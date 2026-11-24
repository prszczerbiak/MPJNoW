# ACI Rules RAG

## Cel projektu

Projekt bada wpływ doboru reguł streszczania na wierność i kompletność notatek
klinicznych generowanych z anglojęzycznych transkrypcji ACI-Bench. Obejmuje
przygotowanie danych, wyszukiwanie reguł, lokalne generowanie notatek,
**ręczną ocenę ich treści** i analizę statystyczną.

Człowiek tworzy wymagane fakty, dzieli notatki na twierdzenia, przypisuje etykiety
i ocenia kompletność. Program przygotowuje formularze, sprawdza ich strukturę
oraz obecność cytatów, a następnie liczy metryki. Nie proponuje ani nie wybiera
ocen merytorycznych. Projekt ma charakter badawczy.

## Metoda

Generator `Qwen/Qwen2.5-3B-Instruct` pracuje w trzech wariantach. Dodatkowy
wariant ekstrakcyjny stanowi punkt odniesienia.

| Wariant | Sposób przygotowania notatki |
|---|---|
| E | Pierwsze dwie i ostatnie dziesięć wypowiedzi, do 500 słów |
| A | Generowanie bez dodatkowych reguł |
| B | Generowanie z czterema stałymi regułami G01–G04 |
| C | Generowanie z czterema regułami dobranymi do konsultacji |

Baza zawiera 12 reguł ogólnych i 48 szczegółowych. Wyszukiwanie wykorzystuje
`sentence-transformers/all-MiniLM-L6-v2`, okna 128 podtokenów z nakładaniem 32
i maksimum podobieństwa cosinusowego dla każdej reguły.

Warianty A/B/C stosują te same ustawienia i ziarna 17, 42, 2026. Budżet wynosi
8192 tokeny: do 7168 na prompt i 1024 na odpowiedź. Oficjalne podziały to
train — 67, valid — 20 oraz test1/test2/test3 — po 40 konsultacji.
Analiza główna obejmuje `test1`: bez wykluczeń powstaje **400 notatek**,
czyli 40 dla E oraz po 120 dla A/B/C.

## Ręczna ocena

Obowiązuje pierwotna procedura z dwiema niezależnymi ocenami i trzecią osobą
rozstrzygającą. Dwóch wykonawców projektu nie zastępuje trzeciej niezależnej
roli bez jawnej zmiany protokołu.

1. Obaj anotatorzy niezależnie tworzą listy faktów z 40 transkrypcji przed
   udostępnieniem notatek. Trzeci anotator uzgadnia końcowe listy.
2. Notatki są zaślepiane i ręcznie dzielone na twierdzenia. Podział sprawdzają
   obaj główni anotatorzy przed niezależnym nadawaniem etykiet.
3. Każdy z nich ocenia wszystkie 400 notatek: etykiety P/S/N, cytaty,
   uzasadnienia oraz odtworzenie każdego wymaganego faktu.
4. Trzeci anotator zapisuje rozstrzygnięcia. Zgodność anotatorów oblicza się
   na ocenach sprzed uzgodnienia.

Pełne formularze, definicje i polecenia zawiera
[instrukcja anotacji](docs/ANNOTATION_GUIDELINES.md).
Walidacja cytatu potwierdza jego obecność, a nie zgodność znaczenia.

## Profil sprzętowy i instalacja

Generator używa kwantyzacji NF4, podwójnej kwantyzacji i obliczeń bfloat16.
Do generowania potrzebna jest karta NVIDIA udostępniona PyTorch przez CUDA.
Ręczna anotacja i obliczanie metryk nie wymagają modelu oceniającego ani GPU.
Wersje i platformę sprawdzonego środowiska opisuje nagłówek `requirements.lock`.

Wymagane są Python 3.11.x (docelowo 3.11.11), Git i zależności projektu.
Polecenia wykonuje się w katalogu `aci-rules-rag`:

```bash
python -m venv .venv
```

Po aktywacji środowiska zainstaluj zależności i sprawdź projekt:

```bash
python -m pip install -r requirements.txt
python scripts/verify_environment.py
```

Do odtworzenia środowiska CUDA użyj indeksu PyTorch zgodnego z wersją CUDA
w nagłówku pliku zależności. Przykładowo dla zapisanego profilu CUDA 12.8:

```bash
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

Testy używają danych syntetycznych, atrap i małej losowej architektury Qwen.
Nie pobierają wag pełnego modelu i nie wykonują badania.

## Uruchomienie potoku

### 1. Dane i konfiguracja

Klonowanie wykonuje się tylko raz, do nieistniejącego katalogu docelowego:

```bash
git clone https://github.com/wyim/aci-bench.git data/raw/aci-bench
git -C data/raw/aci-bench checkout b909b2bb9cf1d19de08df15cddde7bd0179665e4
python -m src.prepare_data
python scripts/verify_tokenizers.py --resolve-revisions
python -m src.configure_run
```

`configure_run` utrwala identyfikator, sumy źródeł i rewizję generatora.
Nie należy wykonywać go ponownie dla zamrożonego, istniejącego przebiegu.

### 2. Generowanie

```bash
python -m src.build_index
python -m src.retrieve_rules --split test1
python -m src.generate_notes plan
python -m src.generate_notes benchmark --execute
python -m src.generate_notes run --execute
```

Plan sprawdza wszystkie budżety; pomiar weryfikuje czas i pamięć najdłuższych
promptów. Wykluczenia dotyczą wspólnie wszystkich wariantów, a ponad 5%
wykluczeń blokuje przebieg. Ukończone zadania nie są powtarzane przy wznowieniu.

**Jeżeli istnieje już kompletny plik `outputs/generations.jsonl`, przejdź do
anotacji.** Zmiana procedury oceny nie wymaga ponownego generowania notatek.

### 3. Fakty i anotacja notatek

Przygotuj puste pakiety list faktów:

```bash
python -m src.prepare_annotation facts
```

Następnie wykonaj ręczne etapy opisane w [instrukcji](docs/ANNOTATION_GUIDELINES.md):
`submit-facts` dla obu anotatorów oraz `freeze-facts` dla każdej konsultacji.
Po zamrożeniu wszystkich list udostępnij zaślepione notatki:

```bash
python -m src.prepare_annotation notes
```

Ludzie przygotowują podziały twierdzeń (`freeze-claims`), niezależne oceny
(`submit-rating`) i decyzje końcowe (`adjudicate`). Nazwy w nawiasach oznaczają
polecenia `python -m src.annotate`; każde przyjmuje ścieżkę ręcznie wypełnionego
pliku JSON. Przykłady wszystkich formatów podaje instrukcja.

### 4. Metryki i analiza

Po ukończeniu niezależnych ocen i wszystkich rozstrzygnięć:

```bash
python -m src.annotate agreement outputs/agreement.json
python -m src.compute_metrics
python -m src.aggregate_results
```

Program oblicza H, H*, C i O z zatwierdzonych ocen, średnie po powtórzeniach
i konsultacjach, zaplanowane testy sparowane, korektę Holma oraz bootstrap.
Niekompletna anotacja blokuje eksport; brakujących ocen nie zastępuje model.

## Struktura projektu

| Element | Zawartość |
|---|---|
| `configs/` | Konfiguracja eksperymentu, generatora i wyszukiwania |
| `data/` | Oficjalne dane, przetworzone konsultacje i manifesty |
| `rules/` | Baza reguł |
| `src/prepare_annotation.py`, `src/annotate.py` | Puste pakiety, zaślepienie, import i kontrola ręcznych ocen |
| `src/compute_metrics.py`, `src/aggregate_results.py` | Obliczenia na zatwierdzonych ocenach |
| `annotations/` | Zgłoszenia, fakty, podziały twierdzeń i rozstrzygnięcia |
| `outputs/` | Wygenerowane notatki i wyniki obliczeń |
| `docs/` | Protokół, instrukcja anotacji i opis implementacji |
| `tests/` | Testy offline i test potoku ręcznej anotacji |

## Dokumentacja i źródła

- [Protokół badawczy](docs/PROTOCOL.md).
- [Instrukcja anotacji](docs/ANNOTATION_GUIDELINES.md).
- [Implementacja i formaty](docs/IMPLEMENTATION.md).
- [Oficjalne dane ACI-Bench](https://github.com/wyim/aci-bench).
- [Generator Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- [Enkoder all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
