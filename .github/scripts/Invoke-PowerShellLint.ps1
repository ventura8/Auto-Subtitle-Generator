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

function Get-FunctionCyclomaticComplexity {
    param([System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst)

    $complexity = 1
    $functionBody = $FunctionAst.Body
    if (-not $functionBody) {
        return $complexity
    }

    $nestedAsts = $functionBody.FindAll(
        {
            param($AstNode)
            $AstNode -is [System.Management.Automation.Language.IfStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.ForEachStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.ForStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.WhileStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.DoWhileStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.DoUntilStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.CatchClauseAst] -or
            $AstNode -is [System.Management.Automation.Language.SwitchStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.TrapStatementAst] -or
            $AstNode -is [System.Management.Automation.Language.BinaryExpressionAst]
        },
        $false
    )

    foreach ($node in $nestedAsts) {
        if ($node -is [System.Management.Automation.Language.IfStatementAst]) {
            # Each if/elseif branch introduces an additional decision path.
            $complexity += $node.Clauses.Count
            continue
        }

        if ($node -is [System.Management.Automation.Language.SwitchStatementAst]) {
            if ($node.Clauses.Count -gt 0) {
                $complexity += $node.Clauses.Count
            }
            else {
                $complexity += 1
            }
            continue
        }

        if ($node -is [System.Management.Automation.Language.BinaryExpressionAst]) {
            $op = $node.Operator.ToString()
            if ($op -in @("And", "Or", "Xor")) {
                $complexity += 1
            }
            continue
        }

        $complexity += 1
    }

    return $complexity
}

function Get-MaximumFunctionNestingDepth {
    param([System.Management.Automation.Language.FunctionDefinitionAst]$FunctionAst)

    $maxDepth = 0

    $null = $FunctionAst.Body.FindAll(
        {
            param($AstNode)
            if (-not ($AstNode -is [System.Management.Automation.Language.StatementAst])) {
                return $false
            }

            $depth = 0
            $parent = $AstNode.Parent
            while ($null -ne $parent) {
                if ($parent -eq $FunctionAst) {
                    break
                }

                if (
                    $parent -is [System.Management.Automation.Language.IfStatementAst] -or
                    $parent -is [System.Management.Automation.Language.ForEachStatementAst] -or
                    $parent -is [System.Management.Automation.Language.ForStatementAst] -or
                    $parent -is [System.Management.Automation.Language.WhileStatementAst] -or
                    $parent -is [System.Management.Automation.Language.DoWhileStatementAst] -or
                    $parent -is [System.Management.Automation.Language.DoUntilStatementAst] -or
                    $parent -is [System.Management.Automation.Language.SwitchStatementAst] -or
                    $parent -is [System.Management.Automation.Language.CatchClauseAst] -or
                    $parent -is [System.Management.Automation.Language.TrapStatementAst]
                ) {
                    $depth += 1
                }

                $parent = $parent.Parent
            }

            if ($depth -gt $maxDepth) {
                $maxDepth = $depth
            }

            return $false
        },
        $false
    )

    return $maxDepth
}

function Test-PowerShellComplexity {
    param(
        [string[]]$Paths,
        [int]$ComplexityLimit,
        [int]$NestingLimit
    )

    $violations = @()

    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) {
            $violations += [pscustomobject]@{
                ScriptName = $path
                Line = 0
                Severity = "Error"
                RuleName = "PSFileNotFound"
                Message = "PowerShell script path was not found."
            }
            continue
        }

        $parseErrors = $null
        $tokens = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$parseErrors)

        if ($parseErrors -and $parseErrors.Count -gt 0) {
            foreach ($parseError in $parseErrors) {
                $violations += [pscustomobject]@{
                    ScriptName = $path
                    Line = $parseError.Extent.StartLineNumber
                    Severity = "Error"
                    RuleName = "PSParserError"
                    Message = $parseError.Message
                }
            }
            continue
        }

        $functions = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
        foreach ($functionAst in $functions) {
            $cyclomatic = Get-FunctionCyclomaticComplexity -FunctionAst $functionAst
            if ($cyclomatic -gt $ComplexityLimit) {
                $violations += [pscustomobject]@{
                    ScriptName = $path
                    Line = $functionAst.Extent.StartLineNumber
                    Severity = "Error"
                    RuleName = "PSCyclomaticComplexity"
                    Message = "Function '$($functionAst.Name)' has cyclomatic complexity $cyclomatic (limit: $ComplexityLimit)."
                }
            }

            $maxDepth = Get-MaximumFunctionNestingDepth -FunctionAst $functionAst
            if ($maxDepth -gt $NestingLimit) {
                $violations += [pscustomobject]@{
                    ScriptName = $path
                    Line = $functionAst.Extent.StartLineNumber
                    Severity = "Error"
                    RuleName = "PSMaximumNestingDepth"
                    Message = "Function '$($functionAst.Name)' has maximum nesting depth $maxDepth (limit: $NestingLimit)."
                }
            }
        }
    }

    return $violations
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
