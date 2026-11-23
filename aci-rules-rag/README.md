# ACI Rules RAG

## Cel projektu

Projekt bada wpływ kontekstowego doboru reguł streszczania na wierność
i kompletność notatek klinicznych generowanych z anglojęzycznych transkrypcji
ACI-Bench. Implementacja obejmuje przygotowanie danych, wyszukiwanie reguł,
lokalne generowanie notatek, automatyczną ocenę wspomaganą audytem oraz analizę
statystyczną.

Zakres pracy dostosowano do dwuosobowego zespołu studentów informatyki.
Przetwarzanie danych i ocena notatek są zautomatyzowane. Ręczna kontrola obejmuje
przypadki nierozstrzygnięte i próbkę kontrolną, a nie wszystkie odpowiedzi.
Projekt jest narzędziem badawczym; nie stanowi systemu oceny dokumentacji
przeznaczonego do zastosowań klinicznych.

## Metoda

Generator `Qwen/Qwen2.5-3B-Instruct` pracuje w trzech wariantach. Dodatkowy wariant
ekstrakcyjny służy jako punkt odniesienia.

| Wariant | Sposób przygotowania notatki |
|---|---|
| E | Pierwsze dwie i ostatnie dziesięć wypowiedzi, z limitem 500 słów |
| A | Generowanie bez dodatkowych reguł |
| B | Generowanie z czterema stałymi regułami G01–G04 |
| C | Generowanie z czterema regułami dobranymi do treści konsultacji |

Baza zawiera 12 reguł ogólnych i 48 szczegółowych. Wariant C wykorzystuje
`sentence-transformers/all-MiniLM-L6-v2`. Wynik reguły jest maksimum podobieństwa
cosinusowego do okien transkrypcji o długości 128 podtokenów i nakładaniu 32.

Warianty A/B/C używają tych samych parametrów generatora oraz ziaren 17, 42 i 2026.
Pełny prompt nie może przekraczać 7168 tokenów, a odpowiedź ma budżet 1024 tokenów.
Oficjalne podziały danych wynoszą: train — 67, valid — 20, test1/test2/test3 —
po 40 konsultacji. Główna analiza porównawcza obejmuje `test1`.

## Automatyczna ocena i zakres pracy ręcznej

Osobny, lokalny weryfikator — domyślnie `Qwen/Qwen2.5-1.5B-Instruct` — wykonuje:

1. Ekstrakcję wymaganych faktów wyłącznie z transkrypcji.
2. Podział wygenerowanej notatki na twierdzenia atomowe.
3. Przypisanie etykiet P — potwierdzone, S — sprzeczne, N — niepotwierdzone.
4. Ocenę poprawnego odtworzenia wymaganych faktów.
5. Ponowną ocenę tych samych twierdzeń bez dostępu do wcześniejszych etykiet.

Program sprawdza strukturę odpowiedzi, kompletność identyfikatorów i obecność
dosłownych cytatów w źródłach. Zgodne, poprawnie zapisane oceny mogą zostać
przyjęte automatycznie. Do kolejki kontroli trafiają:

- rozbieżności między przebiegami i przypadki oznaczone jako niepewne;
- błędny format, nieistniejący cytat, przekroczenie budżetu lub awaria weryfikatora;
- niepuste odpowiedzi sklasyfikowane jako odmowa albo brak treści klinicznej;
- losowe 10% list faktów oraz 10% notatek w każdym wariancie, co najmniej jedna
  pozycja w każdej niepustej grupie.

Każdą pozycję kontroluje jedna z dwóch osób. Identyfikator kontrolera i uzasadnienie
są zapisywane. Wątpliwości wymagające wspólnej konsultacji pozostają w kolejce
do czasu uzgodnienia decyzji. Nie jest wymagana stała trzecia rola anotatora.

Liczba zadań ręcznych zależy od jakości odpowiedzi weryfikatora; 10% określa
wielkość próbki, a nie górny limit kontroli. Próbka obejmuje również ocenę
kompletności podziału na twierdzenia i trafności list wymaganych faktów.

Oceny modelowe są przybliżeniem, nie oceną ekspercką. Dwa przebiegi tego samego
modelu nie są niezależnymi anotatorami. Generator i domyślny weryfikator należą
do tej samej rodziny, co może powodować wspólne błędy. Wyniki nowego trybu mają
oznaczenie `evaluation_method=model_assisted_v2` i charakter eksploracyjny.
Nie są łączone z wynikami pełnej anotacji ręcznej.

## Profil sprzętowy

Generator 3B i weryfikator 1.5B są ładowane w kwantyzacji 4-bitowej NF4
z podwójną kwantyzacją. Precyzja obliczeń pozostaje bfloat16. Profil przygotowano
z myślą o karcie NVIDIA z 4 GiB VRAM; rzeczywiste zużycie pamięci zależy także
od długości wejścia i odpowiedzi. Zmieści się w pamięci dopiero taki przebieg,
który pomyślnie przejdzie pomiar na pełnych promptach. Nie zweryfikowano jeszcze
pełnego wykonania tego profilu na docelowej karcie.

Pakiet bitsandbytes jest częścią zależności projektu. Dostarczony plik blokady
i obraz Docker służą do testów CPU. Uruchomienie modeli wymaga osobnego środowiska
z PyTorch CUDA; w Dockerze konieczne jest również udostępnienie GPU przez
`--gpus all`. Profil nie przełącza się automatycznie na CPU.

Po zmianie modelu należy ponownie wykonać walidację tokenizerów, zbudować indeks,
wykonać wyszukiwanie oraz utworzyć plan i pomiar. Starych wyników, planów ani
zapisów oceny nie należy nadpisywać lub łączyć z nowym przebiegiem.

## Wymagania

- Python 3.11.11 lub zgodna wersja 3.11.x.
- Pakiety z `requirements.lock`; dostarczony plik opisuje sprawdzone środowisko CPU.
- Git do pobrania oficjalnych danych.
- Środowisko GPU zgodne z PyTorch i CUDA do praktycznego wykonania modeli
  z wagami NF4 i obliczeniami bfloat16. Zużycie pamięci określa pomiar przed przebiegiem.
- Dostęp do wag generatora, enkodera i weryfikatora w przypiętych rewizjach.

Modele generujący i oceniający są uruchamiane w oddzielnych etapach. Nie muszą
pozostawać jednocześnie w pamięci GPU. Weryfikator korzysta z istniejących zależności
Transformers i PyTorch; nie wymaga płatnego API.

## Instalacja i testy

Polecenia wykonywane są z katalogu głównego repozytorium.

```bash
python -m venv .venv
```

Po aktywowaniu środowiska wirtualnego:

```bash
python -m pip install -r requirements.lock
python -m pytest -q
```

Alternatywny sposób uruchomienia testów CPU w Dockerze:

```bash
docker build -t aci-rules-rag:cpu .
docker run --rm aci-rules-rag:cpu
```

Testy nie pobierają wag ani nie uruchamiają eksperymentu. Obejmują dane syntetyczne,
atrapy modeli i test zgodności bibliotek. Skrypt `scripts/verify_environment.py`
wykonuje `pip check`, testy i zapisuje dokładne wersje pakietów.
Przed przygotowaniem pliku zależności GPU należy zachować kopię wersji CPU.

## Uruchomienie potoku

### 1. Przygotowanie danych i konfiguracji

Repozytorium danych pobierane jest wyłącznie ze źródła oficjalnego:

```bash
git clone https://github.com/wyim/aci-bench.git data/raw/aci-bench
git -C data/raw/aci-bench checkout b909b2bb9cf1d19de08df15cddde7bd0179665e4
python -m src.prepare_data
python scripts/verify_tokenizers.py
python -m src.configure_run
```

Klonowanie jest potrzebne tylko przy pierwszym uruchomieniu.
`configure_run` uzupełnia identyfikator przebiegu, wersję kodu, sumę reguł,
commit danych i brakujące rewizje generatora oraz weryfikatora.
Nie uruchamia modeli. Parametry badania są zapisane w `configs/`.

### 2. Generowanie

```bash
python -m src.build_index
python -m src.retrieve_rules --split test1
python -m src.generate_notes plan
python -m src.generate_notes benchmark --execute
python -m src.generate_notes run --execute
```

`plan` sprawdza budżety i przygotowuje kolejność zadań. `benchmark` mierzy czas
oraz pamięć dla najdłuższych promptów. `run` wykonuje generowanie i umożliwia
wznowienie po awarii. Przekroczenie budżetu wyklucza konsultację wspólnie
ze wszystkich wariantów. Odsetek wykluczeń większy niż 5% blokuje przebieg.

### 3. Weryfikacja notatek

```bash
python -m src.evaluate_notes facts --execute
python -m src.evaluate_notes review --reviewer student_1
python -m src.evaluate_notes notes --execute
python -m src.evaluate_notes review --reviewer student_2
python -m src.evaluate_notes export
```

Pierwszy etap tworzy listy faktów i kolejkę wymaganych kontroli.
Po ich zatwierdzeniu oceniane są notatki. Polecenie `review` pokazuje transkrypcję,
propozycję i powód skierowania do kontroli. Umożliwia zatwierdzenie,
wczytanie poprawionego JSON lub przerwanie pracy. Obie osoby mogą obsługiwać
kolejkę kolejno, pod własnym identyfikatorem. Jednoczesny zapis nie jest obsługiwany.

Stan pozostałych zadań:

```bash
python -m src.evaluate_notes status
```

Eksport jest blokowany do czasu rozstrzygnięcia wszystkich wymaganych kontroli.
Awarie generatora pozostają w danych jako braki techniczne.
Wyniki oceny znajdują się w `outputs/evaluation/`.

### 4. Analiza

```bash
python -m src.aggregate_results --metrics outputs/evaluation/metrics.csv --evaluation-mode model-assisted --output outputs/evaluation/analysis
```

Obliczane są H, H*, C i O, średnie po powtórzeniach i konsultacjach, sparowane
testy permutacyjne, testy niegorszości kompletności, korekta Holma oraz bootstrap.
Wyniki opisują zgodność z transkrypcją według weryfikatora z audytem człowieka.
Raport obejmuje pochodzenie ocen, liczbę ręcznych kontroli i korekt w losowej próbce.
Nie oblicza się fikcyjnej zgodności między „dwoma anotatorami” na podstawie
dwóch wywołań modelu.

## Struktura repozytorium

| Katalog lub plik | Zawartość |
|---|---|
| `configs/` | Parametry eksperymentu, generatora, wyszukiwania i oceny |
| `data/raw/`, `data/processed/` | Dane oficjalne i dane po walidacji |
| `data/manifests/` | Pochodzenie danych, sumy kontrolne i raporty weryfikacji |
| `rules/` | Baza reguł i jej suma kontrolna |
| `src/prepare_data.py`, `src/configure_run.py` | Przygotowanie danych i konfiguracji |
| `src/build_index.py`, `src/retrieve_rules.py` | Indeksowanie i dobór reguł |
| `src/generate_notes.py`, `src/model_loading.py` | Generowanie i wspólny profil kwantyzacji |
| `src/evaluate_notes.py`, `src/evaluation_prompts.py` | Automatyczna ocena i obsługa kolejki kontroli |
| `src/aggregate_results.py` | Agregacja i analiza statystyczna |
| `src/prepare_annotation.py`, `src/annotate.py`, `src/compute_metrics.py` | Opcjonalny wcześniejszy tryb pełnej anotacji ręcznej |
| `outputs/` | Odpowiedzi modeli, metryki i logi |
| `tests/` | Testy jednostkowe i integracyjne |
| `scripts/` | Weryfikacja środowiska i tokenizerów |
| `docs/` | Protokół, opis oceny i szczegóły implementacji |

Dokumentacja szczegółowa:
[protokół](docs/PROTOCOL.md),
[ocena automatyczna i audyt](docs/AUTOMATIC_EVALUATION.md),
[implementacja](docs/IMPLEMENTATION.md),
[opcjonalna anotacja ręczna](docs/ANNOTATION_GUIDELINES.md).

## Status i ograniczenia

Repozytorium zawiera implementację oraz testy środowiska. Nie zawiera wyników
eksperymentu ani pomiaru pełnego generatora i weryfikatora na docelowym GPU.
Skuteczność automatycznej oceny wymaga sprawdzenia na próbce kontrolnej.
Korekty znalezione w tej próbce należy uwzględnić w interpretacji wyników;
nie stanowią one dowodu poprawności wszystkich pozostałych ocen.

Przed przebiegiem wymagane są wybór sprzętu, przegląd parametrów metody
i dostęp do modeli. Po wykonaniu pozostają kontrola wskazanych przypadków,
ocena jakości audytu oraz interpretacja wyników. Parametrów nie należy
dobierać na podstawie wyników zbioru testowego.

## Źródła

- [ACI-Bench — dane i kod źródłowy](https://github.com/wyim/aci-bench).
- [Qwen2.5-3B-Instruct — generator](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- [Qwen2.5-1.5B-Instruct — domyślny weryfikator](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- [Zheng i in., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
  — omówienie ograniczeń ocen wystawianych przez modele; nie jest walidacją
  kliniczną zastosowanego tu weryfikatora.

Obowiązują licencje danych, bibliotek i modeli wskazane przez ich autorów.
