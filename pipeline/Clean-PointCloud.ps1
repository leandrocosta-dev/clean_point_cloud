<#
Pipeline de limpeza de nuvem de pontos (Revopoint Miraco Plus -> CloudCompare CLI).

Fluxo: N scans PLY (aproximadamente alinhados no referencial do scanner)
       -> ICP sequencial + fusao incremental das clouds
       -> EXTRACT_CC (separa componentes desconexas) -> mantem so UMA
          (a peca; descarta base/mesa/suporte e specks de ruido). Qual
          componente e a peca (a maior ou a menor em pontos) depende da
          geometria do setup de scan -- configuravel em
          keep_largest_component.select ("largest" ou "smallest"). Default
          aqui e "smallest": numa base plana de frente pro scanner a
          amostragem costuma ser mais densa que numa peca em pe vista em
          angulo (foreshortening), entao a peca real pode ter MENOS pontos
          que a base. Confira sempre a listagem de componentes impressa no
          log antes de confiar cegamente.
       -> SOR (outliers estatisticos)
       -> Noise filter (modo RADIUS, equivalente a radius outlier removal)
       -> Crop opcional (remocao de mesa/fundo por caixa delimitadora)
       -> Downsample espacial (voxel-like)
       -> salva nuvem limpa em PLY, pronta para a etapa de fusao de malha.

A etapa "keep_largest_component" roda em 2 chamadas separadas do CloudCompare
porque a decisao de qual componente manter (a de maior numero de pontos) e
feita aqui no PowerShell, entre as duas, lendo o cabecalho PLY de cada
componente exportado.

Sintaxe de todos os comandos CloudCompare verificada contra o codigo-fonte
oficial (qCC/ccCommandLineCommands.cpp e plugins/core/Standard/qRANSAC_SD).
#>

param(
    [string[]]$Scans,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$CCExe,

    [string]$Config,

    [switch]$DryRun,

    # Nao apaga a pasta de staging (_stage_xxxx) com todos os componentes
    # extraidos. Use para inspecionar/trocar manualmente qual componente e a
    # peca quando a escolha automatica (maior/menor) errar.
    [switch]$KeepTemp,

    # Pula a extracao de componentes (estagio 1) e aplica a limpeza final
    # (estagio 2) direto num componente ja extraido manualmente -- usado
    # junto com -KeepTemp de uma rodada anterior, quando a escolha
    # automatica (largest/smallest) errou e voce quer indicar o arquivo certo
    # sem reprocessar a extracao inteira.
    [string]$ResumeComponent
)

$ErrorActionPreference = "Stop"

function Find-CCExe {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) {
            throw "CloudCompare.exe nao encontrado em: $Explicit"
        }
        return (Resolve-Path $Explicit).Path
    }
    $candidates = @(
        "C:\Program Files\CloudCompare\CloudCompare.exe",
        "C:\Program Files (x86)\CloudCompare\CloudCompare.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "CloudCompare.exe nao encontrado nos caminhos padrao. Use -CCExe <caminho>."
}

function Load-Config {
    param([string]$UserConfigPath)
    $defaultPath = Join-Path $PSScriptRoot "config.default.json"
    $cfg = Get-Content $defaultPath -Raw | ConvertFrom-Json
    if ($UserConfigPath) {
        $userCfg = Get-Content $UserConfigPath -Raw | ConvertFrom-Json
        foreach ($section in $userCfg.PSObject.Properties.Name) {
            foreach ($key in $userCfg.$section.PSObject.Properties.Name) {
                $cfg.$section.$key = $userCfg.$section.$key
            }
        }
    }
    return $cfg
}

function Build-IcpFlags {
    param($Icp)
    $flags = @("-ICP", "-REFERENCE_IS_FIRST")
    $flags += @("-MIN_ERROR_DIFF", "$($Icp.min_error_diff)")
    $flags += @("-ITER", "$($Icp.iterations)")
    $flags += @("-OVERLAP", "$($Icp.overlap)")
    $flags += @("-RANDOM_SAMPLING_LIMIT", "$($Icp.random_sampling_limit)")
    if ($Icp.farthest_removal) { $flags += "-FARTHEST_REMOVAL" }
    return $flags
}

function Build-NoiseFlags {
    param($Noise)
    $flags = @("-NOISE")
    if ($Noise.mode -eq "knn") {
        $flags += @("KNN", "$($Noise.value)")
    }
    elseif ($Noise.mode -eq "radius") {
        $flags += @("RADIUS", "$($Noise.value)")
    }
    else {
        throw "noise.mode deve ser 'knn' ou 'radius'"
    }
    $flags += @($Noise.error_type.ToUpper(), "$($Noise.error)")
    if ($Noise.remove_isolated) { $flags += "RIP" }
    return $flags
}

function Build-DownsampleFlags {
    param($Downsample)
    return @("-SS", $Downsample.method.ToUpper(), "$($Downsample.value)")
}

function Build-CropFlags {
    param($Crop)
    $box = ($Crop.box -join ":")
    $flags = @("-CROP", $box)
    if ($Crop.outside) { $flags += "-OUTSIDE" }
    return $flags
}

function Build-LoadAlignFlags {
    param($ScanPaths, $Cfg)
    $flags = @("-O", $ScanPaths[0])
    for ($i = 1; $i -lt $ScanPaths.Count; $i++) {
        $flags += @("-O", $ScanPaths[$i])
        if ($Cfg.icp.enabled) {
            $flags += Build-IcpFlags -Icp $Cfg.icp
        }
        $flags += "-MERGE_CLOUDS"
    }
    return $flags
}

function Build-CleanFlags {
    param($Cfg)
    $flags = @()
    if ($Cfg.sor.enabled) {
        $flags += @("-SOR", "$($Cfg.sor.knn)", "$($Cfg.sor.sigma)")
    }
    if ($Cfg.noise.enabled) {
        $flags += Build-NoiseFlags -Noise $Cfg.noise
    }
    if ($Cfg.crop.enabled) {
        $flags += Build-CropFlags -Crop $Cfg.crop
    }
    if ($Cfg.downsample.enabled) {
        $flags += Build-DownsampleFlags -Downsample $Cfg.downsample
    }
    return $flags
}

function Get-PlyVertexCount {
    param([string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] 8192
        $read = $stream.Read($buffer, 0, $buffer.Length)
        $text = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $read)
        if ($text -match "element vertex (\d+)") {
            return [int]$Matches[1]
        }
        throw "Nao foi possivel ler 'element vertex' do cabecalho PLY: $Path"
    }
    finally {
        $stream.Close()
    }
}

function Invoke-CC {
    param($CCExe, [string[]]$CCArgs, [string]$LogDir, [string]$LogPrefix)

    $stdoutLog = Join-Path $LogDir "$LogPrefix.stdout.log"
    $stderrLog = Join-Path $LogDir "$LogPrefix.stderr.log"

    Write-Host "Comando CloudCompare ($LogPrefix):"
    Write-Host (@($CCExe) + $CCArgs -join " ")

    # CloudCompare.exe e uma aplicacao GUI subsystem: "&" nao captura seu stdout
    # no PowerShell. Start-Process com redirecionamento explicito e necessario.
    # Start-Process -ArgumentList tambem nao coloca aspas automaticamente em
    # elementos com espaco -- sem isso, um nome de arquivo com espaco quebra
    # no meio.
    $quotedArgs = $CCArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $CCExe -ArgumentList $quotedArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    Get-Content $stdoutLog | Write-Host
    if ((Get-Item $stderrLog).Length -gt 0) {
        Get-Content $stderrLog | Write-Host
    }

    if ($p.ExitCode -ne 0) {
        throw "CloudCompare falhou (exit code $($p.ExitCode)) na etapa '$LogPrefix'. Veja $stdoutLog"
    }
}

# --- main ---

$outputDir = Split-Path $Output -Parent
if ($outputDir -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$ccExePath = Find-CCExe -Explicit $CCExe
$cfg = Load-Config -UserConfigPath $Config
$commonFlags = @("-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "PLY")

# --- modo -ResumeComponent: pula a extracao, limpa direto o arquivo indicado ---
# Uso tipico: uma rodada anterior com -KeepTemp errou a escolha automatica
# (largest/smallest); aqui voce aponta manualmente qual *_COMPONENT_*.ply e a
# peca, sem reprocessar o ICP/EXTRACT_CC inteiro de novo.
if ($ResumeComponent) {
    if (-not (Test-Path $ResumeComponent)) {
        Write-Error "Componente nao encontrado: $ResumeComponent"
        exit 1
    }
    $stage2Args = $commonFlags + @("-AUTO_SAVE", "OFF", "-O", (Resolve-Path $ResumeComponent).Path) + `
        (Build-CleanFlags -Cfg $cfg) + @("-SAVE_CLOUDS", "FILE", $Output)

    if ($DryRun) {
        Write-Host (@($ccExePath) + $stage2Args -join " ")
        exit 0
    }
    Invoke-CC -CCExe $ccExePath -CCArgs $stage2Args -LogDir $outputDir -LogPrefix "resume_clean"
    if (-not (Test-Path $Output)) {
        Write-Error "Falha: saida nao gerada em $Output"
        exit 1
    }
    Write-Host "OK: nuvem limpa salva em $Output"
    exit 0
}

if (-not $Scans -or $Scans.Count -eq 0) {
    Write-Error "Informe -Scans (ou use -ResumeComponent para retomar de um componente ja extraido)"
    exit 1
}
foreach ($s in $Scans) {
    if (-not (Test-Path $s)) {
        Write-Error "Scan nao encontrado: $s"
        exit 1
    }
}
$scanPaths = @($Scans | ForEach-Object { (Resolve-Path $_).Path })

if (-not $cfg.keep_largest_component.enabled) {
    $cmdArgs = $commonFlags + @("-AUTO_SAVE", "OFF") + (Build-LoadAlignFlags -ScanPaths $scanPaths -Cfg $cfg) + `
        (Build-CleanFlags -Cfg $cfg) + @("-SAVE_CLOUDS", "FILE", $Output)

    if ($DryRun) {
        Write-Host (@($ccExePath) + $cmdArgs -join " ")
        exit 0
    }
    Invoke-CC -CCExe $ccExePath -CCArgs $cmdArgs -LogDir $outputDir -LogPrefix "clean"
    if (-not (Test-Path $Output)) {
        Write-Error "Falha: saida nao gerada em $Output"
        exit 1
    }
    Write-Host "OK: nuvem limpa salva em $Output"
    exit 0
}

# --- modo keep_largest_component: 2 estagios ---
# Estagio 1 extrai as componentes desconexas num diretorio de staging isolado
# (nunca escreve na pasta original dos scans). Estagio 2 carrega so a maior
# componente (a peca) e aplica a limpeza final.

$stageDir = Join-Path $outputDir "_stage_$([guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

try {
    $stagedScanPaths = @()
    foreach ($s in $scanPaths) {
        $dest = Join-Path $stageDir (Split-Path $s -Leaf)
        Copy-Item $s $dest
        $stagedScanPaths += $dest
    }

    $cc = $cfg.keep_largest_component
    # AUTO_SAVE ON faz o EXTRACT_CC exportar cada componente automaticamente;
    # nao precisa de -SAVE_CLOUDS explicito no final deste estagio.
    $stage1Args = $commonFlags + @("-AUTO_SAVE", "ON") + (Build-LoadAlignFlags -ScanPaths $stagedScanPaths -Cfg $cfg) + `
        @("-EXTRACT_CC", "$($cc.octree_level)", "$($cc.min_points)")

    $firstScanBase = [System.IO.Path]::GetFileNameWithoutExtension($stagedScanPaths[0])
    $stage2OutPlaceholder = Join-Path $stageDir "winner.ply"
    $stage2Args = $commonFlags + @("-AUTO_SAVE", "OFF", "-O", $stage2OutPlaceholder) + (Build-CleanFlags -Cfg $cfg) + `
        @("-SAVE_CLOUDS", "FILE", $Output)

    if ($DryRun) {
        Write-Host "Estagio 1 (extrai componentes):"
        Write-Host (@($ccExePath) + $stage1Args -join " ")
        Write-Host "Estagio 2 (limpeza da maior componente) roda apos escolher o vencedor entre *_COMPONENT_*.ply"
        Write-Host (@($ccExePath) + $stage2Args -join " ")
        exit 0
    }

    Invoke-CC -CCExe $ccExePath -CCArgs $stage1Args -LogDir $stageDir -LogPrefix "stage1_extract"

    $components = Get-ChildItem $stageDir -Filter "$firstScanBase*_COMPONENT_*.ply"
    if (-not $components -or $components.Count -eq 0) {
        Write-Error "EXTRACT_CC nao gerou nenhuma componente (ajuste keep_largest_component.min_points?)"
        exit 1
    }

    $ranked = $components | ForEach-Object {
        [PSCustomObject]@{ Path = $_.FullName; Count = (Get-PlyVertexCount $_.FullName) }
    } | Sort-Object -Property Count -Descending

    Write-Host "Componentes encontradas ($($ranked.Count)), por tamanho:"
    for ($i = 0; $i -lt $ranked.Count; $i++) {
        Write-Host ("  [{0}] {1} pontos - {2}" -f $i, $ranked[$i].Count, $ranked[$i].Path)
    }

    $select = $cc.select
    $winner = if ($select -eq "smallest") { $ranked[$ranked.Count - 1] } else { $ranked[0] }
    Write-Host "Selecao: '$select' -> $($winner.Path) ($($winner.Count) pontos)"
    Write-Host "Se a escolha estiver errada: rode de novo com -KeepTemp, confira a pasta de staging listada abaixo e use -ResumeComponent <caminho> apontando para o componente certo."

    $stage2Args = $stage2Args | ForEach-Object { if ($_ -eq $stage2OutPlaceholder) { $winner.Path } else { $_ } }
    Invoke-CC -CCExe $ccExePath -CCArgs $stage2Args -LogDir $stageDir -LogPrefix "stage2_clean"

    if (-not (Test-Path $Output)) {
        Write-Error "Falha: saida nao gerada em $Output"
        exit 1
    }
    Write-Host "OK: nuvem limpa salva em $Output"
}
finally {
    if ($KeepTemp) {
        Write-Host "Staging mantido (-KeepTemp): $stageDir"
    }
    else {
        Remove-Item -Recurse -Force $stageDir -ErrorAction SilentlyContinue
    }
}

exit 0
