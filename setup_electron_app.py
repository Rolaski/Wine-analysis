"""
Skrypt do automatycznego tworzenia aplikacji Electron dla Wine Analysis.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path


def create_directory_structure():
    """Tworzy strukturę katalogów dla aplikacji Electron."""
    print("Tworzenie struktury katalogów...")

    # Główny katalog projektu
    project_dir = Path("wine-analysis-desktop")
    project_dir.mkdir(exist_ok=True)

    # Katalogi dla aplikacji Streamlit
    streamlit_app_dir = project_dir / "streamlit_app"
    streamlit_app_dir.mkdir(exist_ok=True)

    # Podkatalogi dla aplikacji Streamlit
    for subdir in ["src", "pages", "components", "data"]:
        (streamlit_app_dir / subdir).mkdir(exist_ok=True)

    # Katalog na ikony
    icons_dir = project_dir / "icons"
    icons_dir.mkdir(exist_ok=True)

    return project_dir


def copy_app_files(project_dir):
    """Kopiuje pliki aplikacji do odpowiednich katalogów."""
    print("Kopiowanie plików aplikacji...")

    # Sprawdź, czy pliki aplikacji istnieją
    if not Path("app.py").exists():
        print("Błąd: Plik app.py nie istnieje w bieżącym katalogu.")
        return False

    # Kopiuj główny plik aplikacji
    shutil.copy("app.py", project_dir / "streamlit_app")

    # Kopiuj katalogi
    for subdir in ["src", "pages", "components", "data"]:
        if Path(subdir).exists() and Path(subdir).is_dir():
            # Kopiuj zawartość katalogu
            src_dir = Path(subdir)
            dst_dir = project_dir / "streamlit_app" / subdir

            for item in src_dir.glob("**/*"):
                if item.is_file():
                    # Utwórz podkatalogi, jeśli nie istnieją
                    rel_path = item.relative_to(src_dir)
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)

                    # Kopiuj plik
                    shutil.copy2(item, dst_file)

    return True


def create_icon_file(project_dir):
    """Tworzy przykładową ikonę, jeśli nie istnieje."""
    print("Tworzenie przykładowej ikony...")

    # Tutaj powinno być kopiowanie istniejącej ikony
    # Na potrzeby tego przykładu, sprawdzimy tylko czy ikona istnieje
    icon_path = Path("data") / "wine_icon.ico"
    if icon_path.exists():
        shutil.copy(icon_path, project_dir / "icons" / "wine_icon.ico")
    else:
        print("Uwaga: Nie znaleziono ikony. Aplikacja będzie używać domyślnej ikony.")

    # Przykładowa ikona 16x16 do stworzenia, jeśli nie istnieje w ogóle
    # W praktyce należy użyć prawdziwej ikony


def create_package_json(project_dir):
    """Tworzy plik package.json dla aplikacji Electron."""
    print("Tworzenie pliku package.json...")

    package_data = {
        "name": "wine-analysis-desktop",
        "version": "1.0.0",
        "description": "Aplikacja desktopowa Wine Analysis",
        "main": "main.js",
        "scripts": {
            "start": "electron .",
            "dev": "cross-env DEV_MODE=true electron .",
            "build": "electron-builder",
            "build-win": "electron-builder --win",
            "build-mac": "electron-builder --mac",
            "build-linux": "electron-builder --linux"
        },
        "build": {
            "appId": "com.wineanalysis.app",
            "productName": "Wine Analysis",
            "directories": {
                "output": "dist"
            },
            "files": [
                "main.js",
                "package.json",
                "icons/**/*"
            ],
            "extraResources": [
                {
                    "from": "streamlit_app",
                    "to": "streamlit_app",
                    "filter": ["**/*"]
                },
                {
                    "from": "venv",
                    "to": "venv",
                    "filter": ["**/*"]
                }
            ],
            "win": {
                "target": "nsis",
                "icon": "icons/wine_icon.ico"
            },
            "mac": {
                "target": "dmg",
                "icon": "icons/wine_icon.icns"
            },
            "linux": {
                "target": "AppImage",
                "icon": "icons/wine_icon.png"
            },
            "nsis": {
                "oneClick": False,
                "allowToChangeInstallationDirectory": True,
                "createDesktopShortcut": True,
                "createStartMenuShortcut": True
            }
        },
        "author": "",
        "license": "ISC",
        "devDependencies": {
            "electron": "^27.0.0",
            "electron-builder": "^24.6.4",
            "cross-env": "^7.0.3",
            "wait-on": "^7.0.1"
        }
    }

    with open(project_dir / "package.json", "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2)


def create_main_js(project_dir):
    """Tworzy plik main.js dla aplikacji Electron."""
    print("Tworzenie pliku main.js...")

    main_js_content = """const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');
const url = require('url');
const { spawn } = require('child_process');
const waitOn = require('wait-on');
const isDevMode = process.env.DEV_MODE === 'true';

// Globalny uchwyt do okna i procesu Streamlit
let mainWindow;
let streamlitProcess;

// Funkcja do znalezienia ścieżek Pythona i środowiska wirtualnego
function getPythonPaths() {
  const appPath = app.getAppPath();
  const isPackaged = app.isPackaged;
  let pythonExecutable, streamlitScript, appScriptPath;

  if (isPackaged) {
    // Ścieżki dla spakowanej aplikacji
    const resourcesPath = process.resourcesPath;
    const venvPath = path.join(resourcesPath, 'venv');
    const streamlitAppPath = path.join(resourcesPath, 'streamlit_app');

    if (process.platform === 'win32') {
      pythonExecutable = path.join(venvPath, 'Scripts', 'python.exe');
      streamlitScript = path.join(venvPath, 'Scripts', 'streamlit.exe');
    } else {
      pythonExecutable = path.join(venvPath, 'bin', 'python');
      streamlitScript = path.join(venvPath, 'bin', 'streamlit');
    }

    appScriptPath = path.join(streamlitAppPath, 'app.py');
  } else {
    // Ścieżki dla trybu developerskiego
    const venvPath = path.join(appPath, 'venv');

    if (process.platform === 'win32') {
      pythonExecutable = path.join(venvPath, 'Scripts', 'python.exe');
      streamlitScript = path.join(venvPath, 'Scripts', 'streamlit.exe');
    } else {
      pythonExecutable = path.join(venvPath, 'bin', 'python');
      streamlitScript = path.join(venvPath, 'bin', 'streamlit');
    }

    appScriptPath = path.join(appPath, 'streamlit_app', 'app.py');
  }

  return { pythonExecutable, streamlitScript, appScriptPath };
}

// Tworzymy funkcję do uruchamiania Streamlit
function startStreamlit() {
  // Pobierz ścieżki
  const { pythonExecutable, streamlitScript, appScriptPath } = getPythonPaths();

  console.log(`Python: ${pythonExecutable}`);
  console.log(`Streamlit: ${streamlitScript}`);
  console.log(`App: ${appScriptPath}`);

  // Sprawdź, czy pliki istnieją
  const fs = require('fs');
  if (!fs.existsSync(appScriptPath)) {
    console.error(`Nie znaleziono pliku aplikacji: ${appScriptPath}`);
    app.quit();
    return;
  }

  // Uruchom proces Streamlit
  console.log('Uruchamianie Streamlit...');

  // Używamy streamlit z wirtualnego środowiska
  if (fs.existsSync(streamlitScript)) {
    streamlitProcess = spawn(streamlitScript, [
      'run', 
      appScriptPath, 
      '--server.headless', 'true', 
      '--server.port', '8501', 
      '--browser.serverAddress', 'localhost',
      '--server.enableCORS', 'false', 
      '--server.enableXsrfProtection', 'false'
    ]);
  } else {
    // Fallback - używamy modułu streamlit przez pythona
    streamlitProcess = spawn(pythonExecutable, [
      '-m', 'streamlit', 
      'run', 
      appScriptPath, 
      '--server.headless', 'true', 
      '--server.port', '8501', 
      '--browser.serverAddress', 'localhost',
      '--server.enableCORS', 'false', 
      '--server.enableXsrfProtection', 'false'
    ]);
  }

  // Obsługa wyjścia procesu Streamlit
  streamlitProcess.stdout.on('data', (data) => {
    console.log(`Streamlit stdout: ${data}`);
  });

  streamlitProcess.stderr.on('data', (data) => {
    console.error(`Streamlit stderr: ${data}`);
  });

  streamlitProcess.on('close', (code) => {
    console.log(`Streamlit zakończony z kodem: ${code}`);
  });

  // Czekamy, aż serwer Streamlit będzie dostępny
  const url = 'http://localhost:8501';
  console.log(`Czekam na uruchomienie Streamlit pod adresem: ${url}`);

  waitOn({ resources: [url], timeout: 30000 }).then(() => {
    console.log('Streamlit jest gotowy, otwieranie okna aplikacji...');
    createWindow(url);
  }).catch((err) => {
    console.error('Błąd podczas uruchamiania Streamlit:', err);
    app.quit();
  });
}

// Tworzenie okna aplikacji
function createWindow(streamlitUrl) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'Wine Analysis',
    icon: path.join(app.getAppPath(), 'icons', 'wine_icon.ico'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Ładowanie aplikacji Streamlit
  mainWindow.loadURL(streamlitUrl);

  // Otwórz DevTools w trybie deweloperskim
  if (isDevMode) {
    mainWindow.webContents.openDevTools();
  }

  // Obsługa zamknięcia okna
  mainWindow.on('closed', () => {
    mainWindow = null;

    // Zatrzymaj proces Streamlit
    if (streamlitProcess) {
      console.log('Zatrzymywanie procesu Streamlit...');
      // Na Windows potrzebujemy zabić proces i jego dzieci
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', streamlitProcess.pid, '/f', '/t']);
      } else {
        streamlitProcess.kill();
      }
    }
  });

  // Tworzenie menu aplikacji
  const menuTemplate = [
    {
      label: 'Plik',
      submenu: [
        { role: 'reload', label: 'Odśwież' },
        { type: 'separator' },
        { role: 'quit', label: 'Zakończ' }
      ]
    },
    {
      label: 'Edycja',
      submenu: [
        { role: 'cut', label: 'Wytnij' },
        { role: 'copy', label: 'Kopiuj' },
        { role: 'paste', label: 'Wklej' }
      ]
    },
    {
      label: 'Widok',
      submenu: [
        { role: 'zoomIn', label: 'Powiększ' },
        { role: 'zoomOut', label: 'Pomniejsz' },
        { role: 'resetZoom', label: 'Normalny rozmiar' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Pełny ekran' }
      ]
    },
    {
      label: 'Pomoc',
      submenu: [
        {
          label: 'O aplikacji',
          click: () => {
            const { dialog } = require('electron');
            dialog.showMessageBox(mainWindow, {
              title: 'O Wine Analysis',
              message: 'Wine Analysis v1.0.0',
              detail: 'Aplikacja do analizy i eksploracji zbioru danych Wine Dataset z UCI.',
              icon: path.join(app.getAppPath(), 'icons', 'wine_icon.ico')
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);
}

// Inicjalizacja aplikacji
app.on('ready', startStreamlit);

// Obsługa zamknięcia wszystkich okien
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null && streamlitProcess) {
    createWindow('http://localhost:8501');
  }
});

// Obsługa zamknięcia aplikacji
app.on('will-quit', () => {
  // Zatrzymaj proces Streamlit
  if (streamlitProcess) {
    console.log('Zatrzymywanie procesu Streamlit przed zamknięciem aplikacji...');
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', streamlitProcess.pid, '/f', '/t']);
    } else {
      streamlitProcess.kill();
    }
  }
});
"""

    with open(project_dir / "main.js", "w", encoding="utf-8") as f:
        f.write(main_js_content)


def create_setup_scripts(project_dir):
    """Tworzy skrypty do konfiguracji środowiska."""
    print("Tworzenie skryptów konfiguracyjnych...")

    # Skrypt batch dla Windows
    setup_bat_content = """@echo off
echo Tworzenie wirtualnego środowiska Python...
python -m venv venv
call venv\\Scripts\\activate.bat

echo Instalowanie zależności...
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy plotly mlxtend

echo Instalowanie zależności Node.js...
npm install

echo Środowisko gotowe!
echo Aby uruchomić aplikację, wpisz: npm run dev
pause
"""

    with open(project_dir / "setup_env.bat", "w", encoding="utf-8") as f:
        f.write(setup_bat_content)

    # Skrypt shell dla macOS/Linux
    setup_sh_content = """#!/bin/bash
echo "Tworzenie wirtualnego środowiska Python..."
python3 -m venv venv
source venv/bin/activate

echo "Instalowanie zależności..."
pip install --upgrade pip
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy plotly mlxtend

echo "Instalowanie zależności Node.js..."
npm install

echo "Środowisko gotowe!"
echo "Aby uruchomić aplikację, wpisz: npm run dev"
"""

    with open(project_dir / "setup_env.sh", "w", encoding="utf-8") as f:
        f.write(setup_sh_content)

    # Uprawnienia wykonywania dla skryptu shell na Unix-like
    if os.name != 'nt':  # Unix-like
        os.chmod(project_dir / "setup_env.sh", 0o755)


def create_readme(project_dir):
    """Tworzy plik README.md z instrukcjami."""
    print("Tworzenie pliku README.md...")

    readme_content = """# Wine Analysis Desktop

Aplikacja desktopowa do analizy i eksploracji zbioru danych Wine Dataset z UCI.

## Wymagania

- Node.js
- Python 3.8 lub nowszy
- npm

## Instalacja i uruchomienie

### Konfiguracja środowiska

1. Uruchom skrypt konfiguracyjny:
   - Windows: `setup_env.bat`
   - macOS/Linux: `./setup_env.sh`

2. Skrypt wykona następujące czynności:
   - Utworzy wirtualne środowisko Pythona
   - Zainstaluje wymagane pakiety Pythona
   - Zainstaluje zależności Node.js

### Uruchamianie aplikacji w trybie deweloperskim

```bash
npm run dev
```

### Budowanie aplikacji produkcyjnej

- Dla Windows:
  ```bash
  npm run build-win
  ```

- Dla macOS:
  ```bash
  npm run build-mac
  ```

- Dla Linux:
  ```bash
  npm run build-linux
  ```

Pliki instalacyjne zostaną utworzone w katalogu `dist/`.

## Struktura projektu

- `main.js` - główny plik aplikacji Electron
- `streamlit_app/` - katalog z aplikacją Streamlit
- `venv/` - wirtualne środowisko Pythona
- `icons/` - ikony aplikacji
- `dist/` - skompilowane pliki aplikacji (po zbudowaniu)

## Rozwiązywanie problemów

- Jeśli aplikacja nie uruchamia się, sprawdź logi konsoli
- Upewnij się, że wszystkie zależności są zainstalowane
- Sprawdź, czy porty nie są zajęte przez inne aplikacje

## Licencja

ISC
"""

    with open(project_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)


def setup_electron_project():
    """Główna funkcja konfigurująca projekt Electron."""
    print("========================================")
    print("Konfiguracja projektu Electron dla Wine Analysis")
    print("========================================")

    # Tworzenie struktury katalogów
    project_dir = create_directory_structure()

    # Kopiowanie plików aplikacji
    if not copy_app_files(project_dir):
        print("Błąd: Nie można skopiować plików aplikacji.")
        return

    # Tworzenie ikony
    create_icon_file(project_dir)

    # Tworzenie plików konfiguracyjnych
    create_package_json(project_dir)
    create_main_js(project_dir)
    create_setup_scripts(project_dir)
    create_readme(project_dir)

    print("\nKonfiguracja zakończona pomyślnie!")
    print(f"\nAby kontynuować, przejdź do katalogu '{project_dir}' i uruchom:")
    if os.name == 'nt':  # Windows
        print("  setup_env.bat")
    else:  # Unix-like
        print("  ./setup_env.sh")
    print("\nPo zakończeniu instalacji możesz uruchomić aplikację w trybie deweloperskim:")
    print("  npm run dev")
    print("\nAby zbudować aplikację produkcyjną:")
    print("  npm run build-win (dla Windows)")
    print("  npm run build-mac (dla macOS)")
    print("  npm run build-linux (dla Linux)")


if __name__ == "__main__":
    setup_electron_project()