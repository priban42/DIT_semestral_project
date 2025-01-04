# Estimátor Rychlosti Drážní Soupravy Umístěn ve Vozidle

## Přehled
Tento projekt má za cíl vytvořit estimátor rychlosti drážní soupravy, implementovaný v Pythonu 3.11.
## Struktura Projektu

Projekt obsahuje následující soubory:

- **`video_annotation_tool.py`**: Nástroj pro anotaci snímků ve videu.
- **`audio_utils.py`**: Nástroje pro zpracování zvuku.
- **`utils.py`**: Obecné utility funkce používané v celém projektu.
- **`main.py`**: Příklad použití.
- **`annotated_frames.csv`**: soubor obsahující manuálně anotovaná data z videa.
- **`Labels2.txt`**:  soubor obsahující manuálně anotovaná data z audia.
  
Anotovaná data vycházejí z tohoto videa:
[YouTube Video](https://www.youtube.com/watch?v=L_KOd5PTQPM&t=2307s)

## Instalace

Pro nastavení projektu potřebujete nainstalovaný Python 3.11. Postupujte podle těchto kroků:

   ```bash
   git clone https://github.com/priban42/DIT_semestral_project.git
   cd DIT_semestral_project
   pip install -r requirements.txt
