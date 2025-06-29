# Triple Barrier Application - Build and Run Script
# This script builds and runs the Qt6 C++ application

param(
    [switch]$Clean,
    [switch]$BuildOnly,
    [switch]$RunOnly,
    [string]$BuildType = "Debug",
    [string]$QtPath = ""
)

# Colors for output
$Red = "Red"
$Green = "Green" 
$Yellow = "Yellow"
$Blue = "Blue"

function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Find-QtInstallation {
    # Common Qt installation paths on Windows
    $CommonPaths = @(
        "C:\Qt\*\msvc*",
        "C:\Qt\*\mingw*",
        "$env:USERPROFILE\Qt\*\msvc*",
        "$env:USERPROFILE\Qt\*\mingw*"
    )
    
    foreach ($Path in $CommonPaths) {
        $QtDirs = Get-ChildItem -Path $Path -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if ($QtDirs) {
            return $QtDirs[0].FullName
        }
    }
    return $null
}

function Test-Prerequisites {
    Write-ColorOutput $Blue "Checking prerequisites..."
    
    # Check if CMake is available
    try {
        $cmake = Get-Command cmake -ErrorAction Stop
        Write-ColorOutput $Green "✓ CMake found: $($cmake.Source)"
    }
    catch {
        Write-ColorOutput $Red "✗ CMake not found. Please install CMake and add it to PATH."
        return $false
    }
    
    # Check if Qt6 is available
    if (-not $script:QtPath) {
        Write-ColorOutput $Red "✗ Qt6 installation not found. Please specify -QtPath parameter."
        return $false
    }
    
    Write-ColorOutput $Green "✓ Qt6 found: $script:QtPath"
    return $true
}

# Main script execution
Write-ColorOutput $Blue "=== Triple Barrier Application Build Script ==="

# Set script variables
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ProjectRoot "build"
$Executable = Join-Path $BuildDir "frontend\TripleBarrierApp.exe"

# Find Qt installation if not provided
if (-not $QtPath) {
    Write-ColorOutput $Yellow "Searching for Qt installation..."
    $QtPath = Find-QtInstallation
}
$script:QtPath = $QtPath

# Check prerequisites
if (-not (Test-Prerequisites)) {
    exit 1
}

# Clean build directory if requested
if ($Clean) {
    Write-ColorOutput $Yellow "Cleaning build directory..."
    if (Test-Path $BuildDir) {
        Remove-Item -Path $BuildDir -Recurse -Force
        Write-ColorOutput $Green "✓ Build directory cleaned"
    }
}

# Skip build if RunOnly is specified
if (-not $RunOnly) {
    Write-ColorOutput $Blue "Configuring project with CMake..."
    
    # Create build directory
    if (-not (Test-Path $BuildDir)) {
        New-Item -Path $BuildDir -ItemType Directory -Force | Out-Null
    }
    
    # Configure with CMake
    $ConfigureCmd = "cmake -G `"Ninja`" -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_PREFIX_PATH=`"$QtPath`" -S `"$ProjectRoot`" -B `"$BuildDir`""
    
    Write-ColorOutput $Yellow "Running: $ConfigureCmd"
    
    try {
        Invoke-Expression $ConfigureCmd
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configure failed"
        }
        Write-ColorOutput $Green "✓ Project configured successfully"
    }
    catch {
        Write-ColorOutput $Red "✗ CMake configuration failed: $($_.Exception.Message)"
        exit 1
    }
    
    Write-ColorOutput $Blue "Building project..."
    
    # Build with CMake
    $BuildCmd = "cmake --build `"$BuildDir`" --config $BuildType"
    
    Write-ColorOutput $Yellow "Running: $BuildCmd"
    
    try {
        Invoke-Expression $BuildCmd
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        Write-ColorOutput $Green "✓ Project built successfully"
    }
    catch {
        Write-ColorOutput $Red "✗ Build failed: $($_.Exception.Message)"
        exit 1
    }
}

# Run the application if not BuildOnly
if (-not $BuildOnly) {
    Write-ColorOutput $Blue "Running application..."
    
    if (-not (Test-Path $Executable)) {
        Write-ColorOutput $Red "✗ Executable not found: $Executable"
        Write-ColorOutput $Yellow "Make sure the build completed successfully"
        exit 1
    }
    
    Write-ColorOutput $Green "✓ Starting Triple Barrier Application..."
    Write-ColorOutput $Yellow "Executable: $Executable"
    
    # Set Qt environment and run
    $env:PATH = "$QtPath\bin;$env:PATH"
    
    try {
        & $Executable
    }
    catch {
        Write-ColorOutput $Red "✗ Failed to run application: $($_.Exception.Message)"
        exit 1
    }
}

Write-ColorOutput $Green "=== Script completed successfully! ==="
