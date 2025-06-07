# Jednostavan Video Editor zasnovan na SAM2 modelu

Ovo je jednostavna Python aplikacija koja koristi **SAM2 (Segment Anything Model 2)** razvijen od strane Meta AI tima za segmentaciju objekata na video zapisima. Omogućava korisnicima da pomoću pozitivnih i negativnih tačaka izdvoje objekte iz videa i primene različite vizuelne efekte – kako na objekte, tako i na pozadinu.

## Sadržaj

- [Karakteristike](#Karakteristike)  
- [Potrebne tehnologije i biblioteke](#Potrebne-tehnologije-i-biblioteke)  
- [Instalacija](#Instalacija)  
- [Pokretanje aplikacije](#Pokretanje-aplikacije)  
- [Kreiranje projekta](#kreiranje-projekta)  
- [Pokretanje projekta](#pokretanje-projekta)  

##  Karakteristike

- ✅ Precizna segmentacija objekata u videu korišćenjem SAM2 modela  
- ✅ Selekcija objekta pomoću pozitivnih i negativnih tačaka  
- ✅ Primena različitih efekata na objekat ili pozadinu (npr. zamućenje, isticanje, zamena boje)  
- ✅ Jednostavan interfejs za korišćenje  
- ✅ Mogućnost izvoza editovanog videa u novi fajl

##  Potrebne tehnologije i biblioteke

Ova aplikacija koristi sledeće biblioteke i alate:

- Python 3.x  
- [PyTorch (torch)](https://pytorch.org/)  
- [SAM2](https://github.com/facebookresearch/sam2)  
- OpenCV (`opencv-python`)  
- NumPy  
- ffmpeg (preporučuje se i instalacija `ffmpeg-python`)

---

## Instalacija

1. Kloniraj repozitorijum:
```bash
git clone https://github.com/Darex97/SAM2-Video-Editor.git
cd SAM2-Video-Editor
```

2. Instaliraj zavisnosti:
```bash
pip install -e .
pip install sam2
```

3. Preuzmi pretrenirane modele (checkpoint fajlove) za SAM2 sa zvanične SAM2 stranice:
https://github.com/facebookresearch/sam2

## Pokretanje aplikacije

Nakon što instaliraš sve zavisnosti i preuzmeš SAM2 model, aplikaciju možeš pokrenuti na jedan od sledećih načina:

###  Opcija 1: Pokretanje iz Visual Studio Code-a

1. Otvori folder projekta u **Visual Studio Code-u**  
2. Otvori fajl `video_editor/app.py`  
3. Klikni na **Run** ili pokreni program preko terminala unutar VS Code-a

###  Opcija 2: Pokretanje iz terminala

Ukoliko si u korenskom direktorijumu projekta (`SAM2-Video-Editor`), koristi komandu:

```bash
python video_editor/app.py


