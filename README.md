# Jednostavan Video Editor zasnovan na SAM2 modelu

Savremene metode za obradu video zapisa sve više koriste napredne modele mašinskog učenja kako bi povećale tačnost i brzinu segmentacije sadržaja. U ovom projektu istražuje se primena **SAM2 (Segment Anything Model 2 https://github.com/facebookresearch/sam2)** modela, koji predstavlja unapređenu verziju originalnog SAM modela za segmentaciju slika, sa poboljšanjima u brzini obrade i preciznijoj detekciji objekata.

Cilj ovog rada i aplikacije je kreiranje desktop rešenja koje koristi SAM2 model za generičku segmentaciju video zapisa. Korisniku se pruža mogućnost da izabere određeni vremenski trenutak u video zapisu i pomoću pozitivnih i negativnih tačaka precizno izdvoji željene objekte. U slučaju greške, aplikacija omogućava opoziv poslednje odabrane tačke. Nakon toga, kreiraju se maske za ceo video i omogućava se primena različitih vizuelnih efekata, kako na izdvojene objekte, tako i na pozadinu.

## Sadržaj

- [SAM2 (Segment Anything Model 2)](#sam2-segment-anything-model-2)  
- [Mogućnosti SAM2 modela](#mogućnosti-sam2-modela)  
- [Problemi koje SAM2 rešava](#problemi-koje-sam2-rešava)  
- [Prednosti korišćenja SAM2 modela](#prednosti-korišćenja-sam2-modela)  
- [Konkurentna rešenja](#konkurentna-rešenja)  
- [Ograničenja SAM2 modela](#ograničenja-sam2-modela)  
- [Potrebne tehnologije i biblioteke](#potrebne-tehnologije-i-biblioteke)  
- [Instalacija](#instalacija)  
- [Pokretanje aplikacije](#pokretanje-aplikacije)  
- [Implementacija](#implementacija)  
- [Primeri nekih funkcija za obradu slika](#primeri-nekih-funkcija-za-obradu-slika)
- [Zaključak](#zaključak)
- [Citiranje](#citiranje)

## SAM2 (Segment Anything Model 2)

Segment Anything Model 2 (SAM2) predstavlja napredni model za segmentaciju koji omogućava precizno izdvajanje i praćenje objekata kako na slikama, tako i na video zapisima. Razvio ga je Meta tim kao unapređenu verziju originalnog Segment Anything Modela.

SAM2 korisniku pruža mogućnost da interaktivno izabere objekte pomoću pozitivnih i negativnih tačaka na slici ili u određenom trenutku video zapisa, kao i putem pravougaonika za selekciju. Kao rezultat, model generiše precizne maske koje tačno označavaju poziciju izabranog objekta kroz ceo video ili na pojedinačnoj slici. Pored toga, SAM2 postiže značajno brže i tačnije rezultate u poređenju sa prethodnom verzijom, čineći ga efikasnim alatom za primenu u obradi vizuelnih sadržaja.

## Mogućnosti SAM2 modela

- **Segmentacija objekata na slikama i video zapisima**  
  SAM2 može precizno izdvojiti i pratiti objekte kako na pojedinačnim slikama, tako i kroz čitave video zapise.

- **Interaktivni izbor objekata**  
  Korisnik može selektovati objekte pomoću pozitivnih i negativnih tačaka ili pravougaonika za selekciju, čime se postiže veća preciznost segmentacije.

- **Generisanje maski**  
  Model kreira detaljne maske koje predstavljaju tačne pozicije izdvojenih objekata u svakom kadru video zapisa ili na slici.

- **Poboljšana tačnost i brzina**  
  U poređenju sa prethodnim verzijama, SAM2 donosi značajna poboljšanja u brzini obrade i preciznosti segmentacije.

- **Podrška za različite vrste objekata**  
  Model je generički i sposoban za segmentaciju širokog spektra objekata bez potrebe za dodatnim treninzima ili podešavanjima.

## Problemi koje SAM2 rešava

SAM2 model rešava nekoliko ključnih problema u oblasti segmentacije slike i videa. Prvo, omogućava preciznu segmentaciju objekata čak i u složenim scenama sa više objekata i zahtevnim pozadinama, što je izazov za mnoge tradicionalne metode. Takođe, pruža interaktivnu i fleksibilnu selekciju objekata koristeći različite pristupe, poput tačaka i pravougaonika, čime se poboljšava tačnost segmentacije i korisničko iskustvo. Jedna od važnih karakteristika SAM2 je segmentacija kroz video zapise, gde model omogućava dosledno praćenje i izdvajanje objekata kroz sve kadrove, što značajno olakšava obradu dinamičnih scena. Pored toga, SAM2 je optimizovan za brzinu i efikasnost, što omogućava rad u realnom vremenu ili obradu velikih video fajlova bez gubitka performansi. Model je generički, što znači da može segmentisati širok spektar objekata bez potrebe za dodatnim treninzima ili specijalnim podešavanjima za svaki novi tip objekta, što značajno smanjuje vreme i resurse potrebne za implementaciju. Na kraju, SAM2 unapređuje korisničko iskustvo u video editovanju, omogućavajući preciznu primenu efekata i manipulaciju izdvojenim objektima, čineći proces kreativne obrade videa intuitivnijim i efikasnijim.

## Prednosti korišćenja SAM2 modela

Korišćenje SAM2 modela donosi brojne prednosti u obradi slike i video zapisa. Pre svega, SAM2 omogućava visoku preciznost segmentacije objekata, čak i u složenim scenama, zahvaljujući naprednim algoritmima. Model je izuzetno brz i efikasan, što ga čini pogodnim za rad sa velikim video fajlovima. Takođe, SAM2 je generički i ne zahteva dodatne treninge za nove vrste objekata, što smanjuje vreme i resurse potrebne za prilagođavanje različitim projektima. Fleksibilnost u izboru objekata putem tačaka ili pravougaonika dodatno poboljšava korisničko iskustvo i omogućava preciznu kontrolu nad segmentacijom. Osim toga, SAM2 model omogućava konzistentno praćenje objekata kroz sve kadrove video zapisa, što je ključna funkcionalnost za video editovanje. Sve ove prednosti čine SAM2 moćnim i praktičnim alatom za širok spektar primena u oblasti računarskog vida i video obrade.

## Konkurentna rešenja

Trenutno postoji više rešenja za segmentaciju objekata na slikama, od kojih su najpoznatija **Mask R-CNN**, **DeepLab**, **U-Net** i **YOLACT**. Ovi modeli su pokazali dobre rezultate u preciznoj segmentaciji, ali često zahtevaju značajne resurse za treniranje. Još jedan problem je što svi ovi modeli ne mogu da vrše segmentaciju na video zapisima bez integracije sa posebnim algoritmima za praćenje objekata kroz vreme, što dodatno komplikuje implementaciju i povećava upotrebu resursa (SAM + Xmem++ i SAM + Cutie).

Takođe, mnogi tradicionalni pristupi se oslanjaju na specifične domene ili vrste objekata, što ograničava njihovu primenu u svakodnevnim situacijama. Pored toga, segmentacija video zapisa sa doslednim praćenjem objekata kroz sve kadrove često je izazov zbog problema kao što su promene osvetljenja, pozadine i pomeranja objekata.

U poređenju sa originalnim **Segment Anything Modelom (SAM)**, **SAM2** donosi značajna poboljšanja u brzini obrade i tačnosti segmentacije, kao i bolju podršku za video zapise kroz efikasnije praćenje objekata u vremenskoj dimenziji. Za razliku od SAM-a koji je ograničen na statične slike, SAM2 omogućava pouzdanu segmentaciju i praćenje objekata kroz video zapise. Takodje, SAM2 je optimizovan za bržu i precizniju segmentaciju, što ga čini pogodnijim za primenu u zahtevnim situacijama.

SAM2 model kombinuje visoku tačnost segmentacije sa brzinom i fleksibilnošću načina izbora objekata, omogućavajući primenu u raznim slučajeva bez potrebe za dodatnim treninzima ili podešavanjima. Ovo ga čini konkurentnim i pogodnim za moderne aplikacije iz oblasti video editovanja i računarskog vida.

## Ograničenja SAM2 modela

Iako SAM2 predstavlja značajan napredak u oblasti segmentacije objekata na video zapisima, ovaj model i dalje ima određena ograničenja i prostor za unapređenje. Model je veoma precizan i efikasan pri izdvajanja i praćenju objekata na kratkim video zapisima, naročito kada je praćeni objekat jedini deo scene koji se kreće. Međutim, problemi mogu nastati u situacijama kada je ugao kamere nepovoljan ili kada scena sadrži više sličnih objekata, što može dovesti do smanjenja tačnosti segmentacije i grešaka u prepoznavanju, gde model može pomešati slične objekte sa onima koje treba pratiti.

Takođe, vreme obrade brzo raste ukoliko se prati više objekata simultano, jer se svaki objekat obrađuje zasebno. Ovo može dovesti do opterećenja sistema i mogućih grešaka, posebno na računarima sa slabijom hardverskom konfiguracijom. Još jedna značajna mana je loša predikcija maski kod objekata koji se kreću velikom brzinom, što može rezultirati deformacijama maski ili delimičnim pokrivanjem objekata u pojedinim frejmovima.

Pored toga, iako SAM2 u većini slučajeva generiše precizne maske, neophodno je vršiti proveru kvaliteta tih maski i po potrebi ih ručno korigovati na problematičnim frejmovima dodavanjem dodatnih instrukcija za segmentaciju. Ova dorada omogućava poboljšanje konačnih rezultata i kvalitetniju obradu video sadržaja.


##  Potrebne tehnologije i biblioteke

Ova aplikacija koristi sledeće biblioteke i alate:

- Python 3.x  
- [PyTorch (torch)](https://pytorch.org/)  
- [SAM2](https://github.com/facebookresearch/sam2)  
- OpenCV (`opencv-python`)  
- NumPy  
- ffmpeg

---

## Instalacija

1. Klonirajte repozitorijum:
```bash
git clone https://github.com/Darex97/SAM2-Video-Editor.git
cd SAM2-Video-Editor
```

2. Instalirajte zavisnosti:
```bash
pip install -e .
pip install sam2
```

3. Preuzmite pretrenirane modele (checkpoint fajlove) za SAM2 sa zvanične SAM2 stranice i povežite putanje do preuzetih fajlova u vašoj aplikaciji prilikom kreiranja prediktora:
https://github.com/facebookresearch/sam2

4. (Opciono)
Ukoliko želite da koristite grafičku karticu za ubrzanje rada modela, potrebno je da imate instaliranu CUDA podršku i odgovarajuće drajvere za vašu NVIDIA grafičku kartu. Više informacija o instalaciji CUDA okruženja možete pronaći na zvaničnom sajtu NVIDIA:
https://developer.nvidia.com/cuda-downloads

## Pokretanje aplikacije

Nakon što instalirate sve zavisnosti i preuzmete SAM2 model, aplikaciju možete pokrenuti na jedan od sledećih načina:

###  Opcija 1: Pokretanje iz Visual Studio Code-a

1. Otvorite folder projekta u **Visual Studio Code-u**  
2. Otvorite fajl `videoEditor.py`  
3. Kliknite na **Run** ili pokrenite program preko terminala unutar VS Code-a

###  Opcija 2: Pokretanje iz terminala

Ukoliko ste u korenskom direktorijumu projekta (`SAM2-Video-Editor`), koristite komandu:

```bash
python videoEditor.py
```

## Implementacija

### Kreiranje prediktora

- Originalni kod Meta kompanije je optimizovan za izvršavanje na grafičkim karticama koristeći CUDA.  
- U ovoj implementaciji omogućeno je izvršavanje i na CPU-u, što je sporije u poređenju sa GPU-om.  
- Za pokretanje modela koristimo PyTorch biblioteku za učitavanje prethodno treniranih modela i konfiguracije.  
- Izbor uređaja za izvršavanje (CPU ili GPU) je neophodan za pokretanje predikcija.  
- Prediktor za segmentaciju se kreira pomoću SAM2 modela, prilagođenog za CPU izvršavanje.

```python
device = torch.device("cpu")
output_dir = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\output"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_tiny.pt"
checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_small.pt"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_base_plus.pt"


#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml"
model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_s.yaml"
#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"
#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"


predictor = build_sam2_video_predictor(model_cfg,checkpoint).to(device)
```
### Inicijalizacija prediktora

Nakon kreiranja prediktora, potrebno je da pripremite listu slika tako što podelite video na vremenske intervale i inicijalizujete stanje prediktora na osnovu dobijenih slika. Deljenje videa na frejmove se postiže komandom ffmpeg alata.

```
ffmpeg -i vid.mp4 -q:v 2 -start_number 0 %05d.jpg
```


```python
video_dir = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\video"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path = video_dir)
predictor.reset_state(inference_state)
```
### Kreiranje maski

Glavni razlog za korišćenje ovog prediktora je kreiranje maski za odabrane objekte u svakom delu video zapisa. Kreiranje maski se vrši funkcijom prediktora koja, na osnovu unetih pozitivnih i negativnih tačaka, generiše odgovarajuće maske za svaki od izabranih objekata. Pozitivne tačke označavaju deo ili ceo objekat za koji želite da napravite masku, dok negativne tačke omogućavaju precizno definisanje delova slike koje ne želite da budu uključene u segmentaciju.

```python
prompts={}
    if allChosenObjects != {}:

        for i in range(chosenObj+1):
            if allChosenObjects[chosenObj][1] !=[]:
                points = np.array(allChosenObjects[i][0],dtype=np.float32)
                labels = np.array(allChosenObjects[i][1], np.int32)
                prompts[i]= points,labels
                _,out_obj_ids,out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_from_array,
                    obj_id=i,
                    points=points,
                    labels = labels,)
```

### Propagacija 

Nakon odabira maski za objekte sa slike, potrebno je da izvršite funkciju prediktora koja prolazi kroz sve naredne kadrove video zapisa i pronalazi maske za odabrane objekte na početnoj slici. Povratna vrednost ove funkcije se koristi za obradu svake slike na osnovu dobijenih maski.  
Zbog mogućnosti da korisnik izabere bilo koji vremenski trenutak u video zapisu, predikcija se mora izvršiti u oba smera — od izabranog trenutka do kraja video zapisa, kao i od tog trenutka nazad do početka.

```python
video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i]> 0.0).cpu().numpy()
            for i,out_obj_id in enumerate(out_obj_ids)
        }


    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i]> 0.0).cpu().numpy()
            for i,out_obj_id in enumerate(out_obj_ids)
        }
```

Nakon toga je potrebno implementirati željene funkcije za obradu slika koje će koristiti dobijene maske iz propagacije i tako editovati samo delove slike gde se nalazi izabrani objekat.

##Primeri nekih funkcija za obradu slika

```python
def change_BG_gradient(Img, color_start=(0, 0, 1), color_end=(1, 0, 0)):
    
    image_for_process= np.array(Img)
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]

    new_img= np.zeros((h,w,3),dtype=np.uint8)

    gradient_r = np.linspace(color_start[0], color_end[0], w)
    gradient_g = np.linspace(color_start[1], color_end[1], w)
    gradient_b = np.linspace(color_start[2], color_end[2], w)
    
    
    gradient = np.stack([np.tile(gradient_r, (h, 1)),
                         np.tile(gradient_g, (h, 1)),
                         np.tile(gradient_b, (h, 1))],
                         axis=-1)  

    
    new_img[:,:,:] = image_for_process[:, :, :3] * gradient

    
    
    return new_img
```

```python
def change_BG_color(image,case= None):
    image_for_process= np.array(image)
    n_r=0
    n_g=0
    n_b=0
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]
    new_image= np.zeros((h,w,3),dtype=np.uint8)

    if case=="Gray":

        gray = 0.299 * image_for_process[:, :, 0] + 0.587 * image_for_process[:, :, 1] + 0.144 * image_for_process[:, :, 2]
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        new_image[:, :, 0] = gray  
        new_image[:, :, 1] = gray  
        new_image[:, :, 2] = gray 

    elif case=="White":
        n_r=255
        n_g=255
        n_b=255
        new_image[:,:]=[n_r,n_g,n_b]
    elif case=="Green":
        n_r=0
        n_g=255
        n_b=0
        new_image[:,:]=[n_r,n_g,n_b]

    return new_image
```

Nakon obrade svih frejmova video zapisa ponovo se koristi funkcija ffmpeg-a za spajanje svih dobijenih frejmova u video zapis.

```
ffmpeg -framerate 30 -i s%d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

## Zaključak

SAM2 model predstavlja moćno i fleksibilno rešenje za segmentaciju objekata u video zapisima, omogućavajući precizno izdvajanje i praćenje objekata kroz čitav video. Njegova integracija u aplikacije za video obradu olakšava kreiranje sofisticiranih efekata i poboljšava korisničko iskustvo.

Iako SAM2 donosi značajna unapređenja u odnosu na svog prethodnika, posebno u tačnosti i brzini segmentacije, postoje ograničenja vezana za složene scene, brzinu kretanja objekata i zahteve hardvera. Zbog toga je važno pažljivo birati primenu i biti spreman na dodatne dorade maski u zahtevnijim slučajevima.

Prema tome, SAM2 je snažan alat za istraživanje i razvoj u oblasti računarskog vida i video obrade, sa velikim potencijalom za dalji razvoj i primenu u realnim sistemima.

## Citiranje

Ako koristite SAM 2 ili SA-V skup podataka u svom istraživanju, molim Vas da citirate rad koristeći sledeći BibTeX unos kao sto je zatrazeno i na originalnom repozitorijumu:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```


