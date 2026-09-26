param(
    [Parameter(Mandatory = $true)]
    [string[]]$ScriptPaths,

    [ValidateRange(1, 100)]
    [int]$MaxCyclomaticComplexity = 9,

    [ValidateRange(1, 20)]
    [int]$MaxNestingDepth = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Install PSScriptAnalyzer if not available
if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
    Write-Information "PSScriptAnalyzer was not found. Installing module..."
    $psGallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
    if ($psGallery -and $psGallery.InstallationPolicy -ne "Trusted") {
        Set-PSRepository -Name PSGallery -InstallationPolicy Trusted
    }
    Install-Module PSScriptAnalyzer -Scope CurrentUser -Force -Repository PSGallery
}

# Statements that nest a block: each one adds a decision path and a nesting level.
$script:NestingAstTypes = @(
    [System.Management.Automation.Language.IfStatementAst],
    [System.Management.Automation.Language.ForEachStatementAst],
    [System.Management.Automation.Language.ForStatementAst],
    [System.Management.Automation.Language.WhileStatementAst],
    [System.Management.Automation.Language.DoWhileStatementAst],
    [System.Management.Automation.Language.DoUntilStatementAst],
    [System.Management.Automation.Language.SwitchStatementAst],
    [System.Management.Automation.Language.CatchClauseAst],
    [System.Management.Automation.Language.TrapStatementAst]
)

# Nodes inspected for cyclomatic complexity: the nesting statements plus
# binary expressions, which count only for logical operators.
$script:DecisionAstTypes = $script:NestingAstTypes + @(
    [System.Management.Automation.Language.BinaryExpressionAst]
)

function Test-AstIsAnyType {
    param(
        [System.Management.Automation.Language.Ast]$AstNode,
        [type[]]$Types
    )

    foreach ($type in $Types) {
        if ($AstNode -is $type) {
            return $true
        }
    }
    return $false
}

function Get-DecisionNodeWeight {
    param([System.Management.Automation.Language.Ast]$Node)

    if ($Node -is [System.Management.Automation.Language.IfStatementAst]) {
        # Each if/elseif branch introduces an additional decision path.
        return $Node.Clauses.Count
    }

    if ($Node -is [System.Management.Automation.Language.SwitchStatementAst]) {
        return [Math]::Max($Node.Clauses.Count, 1)
    }

    if ($Node -is [System.Management.Automation.Language.BinaryExpressionAst]) {
        return [int]($Node.Operator.ToString() -in @("And", "Or", "Xor"))
    }

    return 1
}

function Get-FunctionCyclomaticComplexity {
    param([System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst)

    $complexity = 1
    $functionBody = $FunctionAst.Body
    if (-not $functionBody) {
        return $complexity
    }

    $decisionTypes = $script:DecisionAstTypes
    $nestedAsts = $functionBody.FindAll(
        {
            param($AstNode)
            Test-AstIsAnyType -AstNode $AstNode -Types $decisionTypes
        },
        $false
    )

    foreach ($node in $nestedAsts) {
        $complexity += Get-DecisionNodeWeight -Node $node
    }

    return $complexity
}

function Get-StatementNestingDepth {
    param(
        [System.Management.Automation.Language.Ast]$Statement,
        [System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst
    )

    $depth = 0
    $parent = $Statement.Parent
    while ($null -ne $parent -and $parent -ne $FunctionAst) {
        if (Test-AstIsAnyType -AstNode $parent -Types $script:NestingAstTypes) {
            $depth += 1
        }
        $parent = $parent.Parent
    }
    return $depth
}

function Get-MaximumFunctionNestingDepth {
    param([System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst)

    $statements = $FunctionAst.Body.FindAll(
        { param($AstNode) $AstNode -is [System.Management.Automation.Language.StatementAst] },
        $false
    )

    $maxDepth = 0
    foreach ($statement in $statements) {
        $depth = Get-StatementNestingDepth -Statement $statement -FunctionAst $FunctionAst
        $maxDepth = [Math]::Max($maxDepth, $depth)
    }
    return $maxDepth
}

function ConvertTo-LintViolation {
    param(
        [string]$ScriptName,
        [int]$Line,
        [string]$RuleName,
        [string]$Message
    )

    return [pscustomobject]@{
        ScriptName = $ScriptName
        Line = $Line
        Severity = "Error"
        RuleName = $RuleName
        Message = $Message
    }
}

function Get-FunctionViolation {
    param(
        [string]$Path,
        [System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst,
        [int]$ComplexityLimit,
        [int]$NestingLimit
    )

    $line = $FunctionAst.Extent.StartLineNumber
    $name = $FunctionAst.Name

    $cyclomatic = Get-FunctionCyclomaticComplexity -FunctionAst $FunctionAst
    if ($cyclomatic -gt $ComplexityLimit) {
        ConvertTo-LintViolation -ScriptName $Path -Line $line -RuleName "PSCyclomaticComplexity" `
            -Message "Function '$name' has cyclomatic complexity $cyclomatic (limit: $ComplexityLimit)."
    }

    $maxDepth = Get-MaximumFunctionNestingDepth -FunctionAst $FunctionAst
    if ($maxDepth -gt $NestingLimit) {
        ConvertTo-LintViolation -ScriptName $Path -Line $line -RuleName "PSMaximumNestingDepth" `
            -Message "Function '$name' has maximum nesting depth $maxDepth (limit: $NestingLimit)."
    }
}

function Get-ScriptViolation {
    param(
        [string]$Path,
        [int]$ComplexityLimit,
        [int]$NestingLimit
    )

    if (-not (Test-Path $Path)) {
        return ConvertTo-LintViolation -ScriptName $Path -Line 0 -RuleName "PSFileNotFound" `
            -Message "PowerShell script path was not found."
    }

    $parseErrors = $null
    $tokens = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$parseErrors)

    if ($parseErrors -and $parseErrors.Count -gt 0) {
        return $parseErrors | ForEach-Object {
            ConvertTo-LintViolation -ScriptName $Path -Line $_.Extent.StartLineNumber -RuleName "PSParserError" -Message $_.Message
        }
    }

    $functions = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
    foreach ($functionAst in $functions) {
        Get-FunctionViolation -Path $Path -FunctionAst $functionAst -ComplexityLimit $ComplexityLimit -NestingLimit $NestingLimit
    }
}

function Test-PowerShellComplexity {
    param(
        [string[]]$Paths,
        [int]$ComplexityLimit,
        [int]$NestingLimit
    )

    $violations = foreach ($path in $Paths) {
        Get-ScriptViolation -Path $path -ComplexityLimit $ComplexityLimit -NestingLimit $NestingLimit
    }
    return @($violations)
}

# Run analysis on all script paths
$issues = foreach ($scriptPath in $ScriptPaths) {
    Invoke-ScriptAnalyzer -Path $scriptPath -Severity Warning,Error -Recurse:$false
}

$complexityIssues = Test-PowerShellComplexity -Paths $ScriptPaths -ComplexityLimit $MaxCyclomaticComplexity -NestingLimit $MaxNestingDepth

if ($complexityIssues) {
    $issues = @($issues) + @($complexityIssues)
}

# Report and fail if issues found
if ($issues) {
    $issues |
        Select-Object ScriptName, Line, Severity, RuleName, Message |
        Format-Table -AutoSize |
        Out-String |
        Write-Output
    throw "PowerShell lint failed. Resolve all PSScriptAnalyzer warnings/errors."
}
