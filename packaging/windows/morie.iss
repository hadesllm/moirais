; Inno Setup script for the morie click-through installer (Windows).
;
; Build (on Windows, with Inno Setup 6 installed):
;   iscc /DMyAppVersion=0.7.2 packaging\windows\morie.iss
;
; Expects the PyInstaller bundle at dist\morie\ (repo root). CI builds
; that bundle on a windows runner first, then runs this script.
;
; Produces: dist\installer\morie-setup.exe  -- a per-user installer that
; needs no admin rights, no Python, no winget.

#ifndef MyAppVersion
  #define MyAppVersion GetEnv("MORIE_VERSION")
#endif
#if MyAppVersion == ""
  #define MyAppVersion "0.0.0"
#endif

#ifndef MorieBundleDir
  #define MorieBundleDir "..\..\dist\morie"
#endif

#define MyAppName "morie"
#define MyAppPublisher "Vansh Singh Ruhela"
#define MyAppURL "https://github.com/hadesllm/morie"
#define MyAppDocsURL "https://hadesllm.github.io/morie/"

[Setup]
; AppId must stay constant across versions so upgrades replace cleanly.
AppId={{A7F3C2E1-9B4D-4E6A-8C1F-2D5E7A9B3C4D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\morie
DefaultGroupName=morie
DisableProgramGroupPage=yes
; Per-user install: no UAC prompt, no admin rights needed.
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=morie-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Tells Windows the installer changed PATH, so new terminals pick it up.
ChangesEnvironment=yes
UninstallDisplayName=morie {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add morie to PATH (lets you type 'morie' in any terminal)"; GroupDescription: "Integration:"

[Files]
Source: "{#MorieBundleDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "morie-console.cmd"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\morie Console"; Filename: "{app}\morie-console.cmd"; Comment: "Open a terminal ready to run morie"
Name: "{group}\morie Documentation"; Filename: "{#MyAppDocsURL}"
Name: "{group}\Uninstall morie"; Filename: "{uninstallexe}"

[Registry]
; Append the install dir to the per-user PATH (only if not already there).
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath('{app}')

[Run]
Filename: "{app}\morie-console.cmd"; Description: "Open morie Console now"; \
  Flags: postinstall skipifsilent nowait unchecked

[Code]
function NeedsAddPath(Param: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  { True only when the install dir is not already on PATH. }
  Result := Pos(';' + ExpandConstant(Param) + ';', ';' + OrigPath + ';') = 0;
end;
