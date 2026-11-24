# Implementacja i formaty danych

Polecenia uruchamia się z katalogu głównego projektu przez `python -m src.MODULE`.
Import modułów nie uruchamia generowania. Pobieranie modeli następuje dopiero
przy wywołaniu poleceń indeksowania, walidacji tokenizerów lub inferencji.
Projekt nie korzysta z płatnych API.

Fakty, podział twierdzeń i oceny przygotowują ludzie. Program obsługuje
formularze, zaślepienie, walidację oraz obliczenia. Procedurę opisuje
[instrukcja anotacji](ANNOTATION_GUIDELINES.md).

Moduł `model_loading.py` ustala parametry NF4, podwójnej kwantyzacji, typu obliczeń
i urządzenia dla generatora. Brak CUDA zatrzymuje inferencję przed pobraniem wag.

## Moduły

| Moduł | Odpowiedzialność |
|---|---|
| `configure_run.py` | Automatyczne uzupełnianie identyfikatora przebiegu, rewizji i sum źródeł |
| `common.py` | Atomowy zapis UTF-8, SHA-256, deterministyczne sumy JSON i ochrona zatwierdzonych plików |
| `prepare_data.py` | Wyszukiwanie CSV, kontrola schematu i integralności źródeł, podziałów, identyfikatorów i duplikatów; oddzielenie referencji |
| `build_index.py` | Walidacja 60 reguł, limity tokenizerów, rewizja enkodera, zapis wektorów i metadanych |
| `retrieve_rules.py` | Okna podtokenów, podobieństwa znormalizowanych wektorów, ranking wszystkich reguł i indeks okien |
| `generate_notes.py` | Wariant E, wspólne prompty, wykluczenia, losowanie kolejności, środowisko, pomiar wydajności i wznawianie generowania |
| `prepare_annotation.py` | Pakiety do tworzenia faktów i udostępnianie zaślepionych notatek po zamrożeniu list |
| `annotate.py` | Niezależne zgłoszenia, kontrola ręcznego podziału twierdzeń, rozstrzygnięcia, zgodność surowa, kappa i macierz niezgodności |
| `compute_metrics.py` | Metryki z uzgodnionych etykiet P/S/N i ocen odtworzenia faktów |
| `aggregate_results.py` | Kompletność zadań, progi braków, średnie po powtórzeniach i konsultacjach, cztery testy oraz bootstrap |

## Pliki danych

`data/raw/aci-bench` jest kopią oficjalnego repozytorium. Dane kliniczne nie są
przepisywane. Importer znajduje pełne pliki CSV po nazwie i schemacie, bez
założenia ich położenia w katalogach. Porównanie z `HEAD` dopuszcza różnice
końców linii CRLF/LF. Lokalne tłumaczenia głównych plików `README.md` i `SETUP.md`
są wyłączone z tej kontroli; pozostałe śledzone pliki nadal jej podlegają.

SHA-256 dotyczy rzeczywistych bajtów użytego źródła. Manifest zawiera również
sumy obu plików przetworzonych. Błędy UTF-8, NUL, znaki zastępcze, puste wymagane
pola, powtórzone identyfikatory i niezgodne liczebności podziałów przerywają import.

`consultations.jsonl` zawiera `encounter_id`, `split`, `source_type` i `transcript`.
`references.jsonl` nie jest odczytywany podczas budowania promptów.

`rule_index.json` przechowuje uporządkowane reguły, wektory jednostkowe, rzeczywiste
długości w tokenach, sumę źródła i metadane przypiętych rewizji enkodera oraz
tokenizera. `window_index.jsonl` zawiera identyfikatory konsultacji i okna,
dokładne identyfikatory tokenów, tekst źródłowy, granice okna w tokenach i wektor.

`retrieval.jsonl` obejmuje wszystkie 60 par konsultacja–reguła, z polami:
`encounter_id`, `rule_id`, `rank`, `similarity_score`, `selected_window_id`,
`window_text`, `selected`, `retrieval_time_s` oraz granicami okna i liczbami tokenów.
Towarzyszący manifest wiąże wynik z rekordami wejściowymi, regułami i indeksem.

`plan.json` zapisuje wylosowaną kolejność wykonania i kompletne prompty.
Budżet jest sprawdzany tokenizerem generatora z `add_special_tokens=False`
po zastosowaniu szablonu rozmowy. Wejście nie jest przycinane. Plan zawiera
manifesty danych, wyszukiwania i konfiguracji. `exclusions.json` przechowuje
wspólne wykluczenia i ich odsetek. Zatwierdzonych plików nie można nadpisać
inną treścią.

`generations.jsonl` obejmuje każde zaplanowane zadanie, także awarie techniczne:

```text
run_id, encounter_id, split, source_type, variant, seed, model, model_revision,
prompt_hash, retrieved_rules, retrieval_scores, prompt, response, input_tokens,
output_tokens, generation_time_s, status, code_version,
task_id, execution_position, environment_id, attempt_count (A/B/C)
```

Wariant E ma `seed=null`, `model=extractive`, `model_revision=null`, pusty prompt,
zero tokenów wejściowych, zmierzoną liczbę tokenów wyjściowych i metadane ekstrakcji.
Dla A/B/C długość wyjścia obejmuje wygenerowane identyfikatory tokenów, w tym EOS,
jeśli wystąpił. Czas E wynosi zero, ponieważ nie wywołuje on modelu; ekstrakcja
jest deterministyczną częścią planowania.

Dopuszczalne statusy to `ok` i `technical_failure`. Odmowa modelu jest technicznie
poprawnie wykonaną odpowiedzią (`ok`); jej treść klasyfikują później anotatorzy.

## Zapis, awarie i wznowienia

Zadania mają stałe identyfikatory. Próba jest zapisywana przed wywołaniem modelu.
Dopuszczalne są najwyżej trzy próby łącznie, także gdy któraś została przerwana.
Wyjątki `RuntimeError` i `OSError` są traktowane jako awarie infrastruktury.
Błędy programistyczne i naruszenia protokołu przerywają wykonanie.

Błędy infrastruktury trafiają do `outputs/logs/infrastructure/`. Uwagi dotyczące
pustej odpowiedzi lub długości trafiają do `outputs/logs/content/` i nie powodują
ponowienia. Wznowienie korzysta z `outputs/tasks/`, a zbiorczy JSONL jest
odtwarzany z tych zapisów.

Plan i środowisko są utrwalane przed wykonaniem zadań. Identyfikator środowiska
uwzględnia biblioteki, Pythona, CUDA, właściwości GPU, konfigurację ładowania
i SHA-256 modułów implementacji. Zmiana kodu — również komentarzy — lub
środowiska wymaga nowego katalogu przebiegu.

Zapis korzysta z pliku tymczasowego w tym samym katalogu, `flush/fsync`
i atomowego `os.replace`. Wyniki JSONL/CSV są deterministyczne albo można je
odtworzyć z zapisów źródłowych. Ponowne zgłoszenie innej treści do zatwierdzonego
pliku jest odrzucane.

Obsługiwany jest jeden proces zapisujący do katalogu przebiegu lub anotacji.
Nie ma synchronizacji rozproszonej ani transakcji obejmującej kilka plików.
Jeśli etap zostanie przerwany w trakcie zapisu wielu plików, należy powtórzyć go
z tymi samymi wejściami. Zatwierdzonych plików nie należy edytować ręcznie.

## Pełna anotacja ręczna

`annotations/private/` jest dostępny tylko koordynatorowi. Anotator otrzymuje
swój katalog pakietów. Listy wymaganych faktów są zamrażane przed udostępnieniem
notatek, a ich sumy kontrolne są ponownie sprawdzane przy obliczaniu metryk.

Niezależne zgłoszenia są przechowywane oddzielnie od decyzji trzeciego anotatora.
Podział na twierdzenia zatwierdzają ludzie; automatyczny podział na zdania nie
jest uznawany za wyznaczenie twierdzeń atomowych. Każdy odtworzony fakt musi
odwoływać się do co najmniej jednego potwierdzonego twierdzenia. Program nie
sprawdza, czy anotatorzy znaleźli wszystkie twierdzenia ani czy ich ocena
kliniczna jest poprawna.

Metryki są powiązane z dokładną wersją udostępnionych odpowiedzi i list faktów.
Awarie pozostają rekordami bez ocen klinicznych. Agregacja wymaga wszystkich
zaplanowanych zadań i właściwych zestawów ziaren. Odrzuca duplikaty, nieznane
statusy, mieszanie środowisk lub przebiegów, dane spoza `test1`, nieprawidłowe
wartości metryk oraz ponad 5% awarii w dowolnym wariancie.

Najpierw uśredniane są kompletne grupy powtórzeń. Cztery testy obejmują wspólne
kompletne konsultacje A/B/C. Niepełne grupy pozostają w raporcie braków.

## Pochodzenie ocen ręcznych

Metryki powstają wyłącznie z zatwierdzonych rozstrzygnięć w katalogu
`adjudicated`. Pole `evaluation_method` ma wartość `human`. Agregacja
odrzuca inne pochodzenie ocen. Sumy kontrolne wiążą oceny z listami
faktów i udostępnionymi notatkami.

## Zakres weryfikacji i ograniczenia

`tests/test_integration.py` sprawdza import CSV, okna tokenów, ranking, planowanie,
atrapę generatora, niezależne listy faktów, zaślepienie notatek, podział na
twierdzenia, niezależne oceny, rozstrzygnięcia, zgodność, metryki i agregację
dwupoziomową. Atrapy generatora i wektorów występują tylko w testach.

Oddzielny test zgodności bibliotek tworzy małą architekturę Qwen z losowymi
wagami i wykonuje obliczenia na CPU. Nie ładuje pełnych wag generatora ani nie generuje notatek.
Testy statystyczne porównują implementację z ustalonymi funkcjami SciPy
i statsmodels na tablicach syntetycznych. Dane testowe trafiają do katalogów
tymczasowych pytest, nie do wyników eksperymentu.

`requirements.lock` powstaje po przejściu `pip check` i wszystkich testów.
Sprawdzone środowisko to Python 3.11.11 na CPU w Linux/Docker. Nie potwierdza to,
że pełny model Qwen zmieści się na docelowym GPU.

Przed utworzeniem pliku zależności dla GPU zachowaj wersję CPU jako
`requirements.cpu.lock`. Dobierz PyTorch zgodny z CUDA i sprzętem, uruchom
weryfikację i używaj jednego środowiska przez cały eksperyment główny.
Nie łącz wyników CPU i GPU.

Przypięty obraz Dockera służy do testów CPU. Wykonanie na GPU wymaga zgodnego
środowiska CUDA i jawnego udostępnienia karty. Domyślny profil wykorzystuje 4-bitowe wagi NF4 i obliczenia bfloat16.
`bitsandbytes` należy do zależności; testy CPU nie potwierdzają działania jego jąder CUDA.
Repozytorium nie zawiera wag modeli ani wygenerowanych wyników eksperymentu.

## Przygotowanie eksperymentu

Potwierdź decyzje z `PROTOCOL.md`, przypisz dwóch anotatorów i trzecią osobę rozstrzygającą, przejrzyj
reguły, uruchom `python -m src.configure_run`, wybierz nowy katalog wyników
oraz utrwal konfigurację i sumy źródeł.

Rewizje modeli można ustalić narzędziem walidacji tokenizerów. Po zamrożeniu nie
należy ich zmieniać. Zamroź ręcznie uzgodnione listy faktów przed oceną notatek.
Sprawdź GPU, CUDA, typ obliczeń i sposób ładowania, wykonaj pomiar na najdłuższych
promptach i przejrzyj raport czasu oraz pamięci przed `run --execute`.
Tych wywołań generatora nie wykonano podczas przygotowania repozytorium.
