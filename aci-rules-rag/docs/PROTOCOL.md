# Protokół badawczy — wersja 3

Repozytorium zawiera środowisko do przeprowadzenia badania, bez wyników i wniosków
eksperymentalnych. Analiza główna obejmuje oficjalny zbiór `test1`. Zbiory `train`
i `valid` służą do kontroli technicznej i dopracowania protokołu przed jego
zamrożeniem. `test2` i `test3` są przeznaczone do osobno zadeklarowanych analiz
dodatkowych. Nie wolno dobierać parametrów na zbiorach testowych, łączyć podziałów
ani stosować walidacji krzyżowej.

## Wersja metody oceny

Podstawowym trybem projektu jest ocena modelowa z audytem dwóch studentów,
oznaczona jako model_assisted_v2. Zastępuje ona pierwotną pełną anotację ręczną.
Zmiana dotyczy źródła list faktów, podziału na twierdzenia i etykiet, dlatego
wyniki mają charakter eksploracyjny. Wzory metryk i porównania wariantów pozostają
bez zmian. Pełna anotacja ręczna jest dostępna wyłącznie jako tryb opcjonalny
i nie może być łączona z ocenami modelowymi.

## Dane i pochodzenie

Jedynym źródłem danych jest [oficjalne repozytorium ACI-Bench](https://github.com/wyim/aci-bench).
Importer wyszukuje znane nazwy plików CSV, sprawdza kolumny i jednoznaczność
dopasowania, zapisuje commit Git oraz SHA-256 każdego użytego pliku.
Wymagane liczebności `train/valid/test1/test2/test3` wynoszą `67/20/40/40/40`.
Identyfikatory konsultacji muszą być unikalne w całym zbiorze.

Dokładne duplikaty transkrypcji i notatek referencyjnych są wykrywane po normalizacji
Unicode NFC i końców linii do LF. Ich obecność wstrzymuje przygotowanie danych
do czasu ręcznej kontroli. Rekordy nie są automatycznie usuwane. Podobieństwo
tekstów nie służy do zmiany próby badawczej. Poza Unicode i końcami linii nie wolno
modyfikować treści klinicznej. Typ źródła pochodzi z oficjalnej kolumny `dataset`.
Notatki referencyjne są zapisywane osobno i nie trafiają do promptów.

## Warianty

- **E:** pierwsze dwie i ostatnie dziesięć wypowiedzi, bez powtórzeń, w kolejności
  źródłowej i z oznaczeniami rozmówców. Rozmowy krótsze niż dwanaście wypowiedzi
  są zachowywane w całości. Przy przekroczeniu 500 słów, liczonych po białych
  znakach, usuwane są całe wypowiedzi od końca, z zachowaniem pierwszej.
  Jeśli sama pierwsza wypowiedź przekracza limit, pozostaje jej pierwsze 500 słów,
  a metadane otrzymują `single_utterance_truncated=true`. Oznaczenie skrócenia
  nie jest dopisywane do tekstu klinicznego. E służy wyłącznie jako opisowy punkt
  odniesienia.
- **A:** wspólne polecenie i pusty blok reguł.
- **B:** dokładnie reguły G01–G04 o brzmieniu określonym w protokole.
- **C:** dokładnie cztery reguły dobrane z zatwierdzonej bazy 60 reguł.

Wspólne polecenie znajduje się w `src/generate_notes.py`. Warianty A/B/C mają
identyczne polecenie, szablon rozmowy, transkrypcję, tokenizer, model, parametry
dekodowania i budżet. Różnią się tylko treścią bloku reguł przed transkrypcją.
Nagłówki notatki to `History`, `Examination`, `Results` oraz `Assessment and plan`.
Transkrypcja jest materiałem źródłowym, a nie instrukcją zmieniającą zadanie.

## Wyszukiwanie reguł

Domyślny enkoder to `sentence-transformers/all-MiniLM-L6-v2` z przypiętą rewizją.
Konfigurację można zmienić przed zamrożeniem protokołu. Okna obejmują 128 podtokenów
treści i nakładają się na 32 podtokeny. Tokeny specjalne enkodera nie wchodzą
do długości okna. Enkoder otrzymuje identyfikatory tokenów bez pośredniego
dekodowania i ponownej tokenizacji. Pozycje w oryginalnym tekście pozostają
dostępne do kontroli.

Każda reguła może mieć najwyżej 96 tokenów generatora i 128 podtokenów treści
enkodera. Baza obejmuje 12 reguł ogólnych i 48 szczegółowych z ośmiu obszarów.
Nie zawiera faktów o pacjentach, przykładów testowych ani porad terapeutycznych.

Wektory są normalizowane. Wynik reguły to największe podobieństwo cosinusowe
do dowolnego okna rozmowy. Wybierane są cztery najlepsze reguły. Remisy rozstrzyga
rosnący identyfikator reguły; przy jednakowych wynikach okien wybierane jest
najwcześniejsze. Zapis obejmuje wszystkie 60 pozycji rankingu, okna, wyniki,
liczby tokenów i czas wyszukiwania. Stosowane jest bezpośrednie porównanie macierzy,
bez LLM tworzącego zapytania i bez rerankera.

## Generator, budżet i wykonanie

Generator to wyłącznie `Qwen/Qwen2.5-3B-Instruct`. Model i tokenizer muszą mieć tę
samą, niezmienną rewizję. Wersja 3 zastępuje generator 7B modelem 3B ze względu
na docelową kartę NVIDIA RTX 3050 Ti Laptop z 4 GiB VRAM. Wagi są ładowane
w 4 bitach (NF4, podwójna kwantyzacja), a obliczenia wykorzystują bfloat16.
Wszystkie warianty A/B/C mają tę samą konfigurację kwantyzacji i ładowania.

Weryfikator to osobny Qwen2.5-1.5B-Instruct, również w NF4 z obliczeniami bfloat16.
Modele pracują kolejno i nie muszą jednocześnie zajmować pamięci GPU. Do ładowania
służy bitsandbytes. Profil wymaga PyTorch z CUDA i jawnie wybiera cuda:0;
nie przełącza się automatycznie na CPU ani na inną precyzję.

Zmiana modelu lub kwantyzacji wymaga nowego przebiegu, ponownej walidacji
rzeczywistego tokenizera, przebudowania indeksu i ponownego pomiaru pamięci.
Nie wolno łączyć wyników poprzedniego profilu 7B z nowym profilem 3B.
Dopasowanie modelu do 4 GiB wymaga pomiaru na pełnych promptach; sam rozmiar wag
nie określa całego zużycia pamięci. Fine-tuning i zmiany tokenizera pozostają
niedozwolone.

Parametry generowania:

| Parametr | Wartość |
|---|---:|
| `do_sample` | `true` |
| `temperature` | 0.2 |
| `top_p` | 0.9 |
| `top_k` | 0 |
| `repetition_penalty` | 1.0 |
| `max_new_tokens` | 1024 |
| `batch_size` | 1 |

Łączny budżet wynosi 8192 tokeny: 1024 na odpowiedź i najwyżej 7168 na pełne
wejście po zastosowaniu szablonu rozmowy. Wejście nie jest przycinane.
Przed wykonaniem sprawdzane są wszystkie prompty A/B/C. Przekroczenie limitu
wyklucza konsultację wspólnie ze wszystkich czterech wariantów; zapis obejmuje
powód i liczby tokenów. Odsetek wykluczeń większy niż 5% blokuje generowanie
i wymaga zmiany protokołu.

A/B/C mają po jednym wywołaniu dla ziaren `17, 42, 2026`, ustawianych ponownie
przed każdym wywołaniem. E wykonuje się raz. Ziarno kolejności zadań to `31415`,
kolejności anotacji — `27182`, a resamplingu — `20261021`.

Przed przebiegiem należy utrwalić kolejność wykonania, wersje danych, reguł
i konfiguracji, wersję kodu, Pythona, bibliotek i CUDA, dostępne GPU, precyzję
oraz sposób ładowania modelu. Pomiar infrastruktury obejmuje najdłuższy pełny
prompt każdego wariantu generatywnego. Jego odpowiedzi są odrzucane i nie stanowią
obserwacji eksperymentalnych. Raport zawiera czas i szczytowe zużycie pamięci GPU.
Pomiar wymaga osobnego uruchomienia; nie wykonano go podczas przygotowania repozytorium.

Awaria infrastruktury dopuszcza najwyżej dwa ponowienia po pierwszej próbie,
z tym samym wejściem i ziarnem. Próba jest zapisywana przed wywołaniem modelu,
więc przerwane wywołanie również zużywa limit. Odmowa, niepotwierdzona treść
i niewłaściwa długość odpowiedzi nie uzasadniają ponowienia. Ukończone odpowiedzi
nie są zmieniane przy wznowieniu. Katalog przebiegu obsługuje jeden proces zapisujący.

## Ocena modelowa i audyt

Lokalny weryfikator Qwen/Qwen2.5-1.5B-Instruct tworzy wymagane fakty wyłącznie
z transkrypcji. Nie otrzymuje notatek referencyjnych ani wygenerowanych na tym
etapie. Lista zawiera identyfikatory, kategorie, treść, dokładne fragmenty źródła,
uzasadnienia i znaczniki niepewności. K musi być dodatnie.

Po rozstrzygnięciu kontroli list faktów weryfikator dzieli notatki na twierdzenia
atomowe, przypisuje P/S/N i ocenia odtworzenie każdego faktu. Drugi przebieg
ocenia ten sam podział bez dostępu do pierwszych etykiet. Nie jest to drugi
niezależny anotator. Zgodność obu przebiegów nie gwarantuje poprawności oceny.

Do człowieka trafiają rozbieżności, niepewność, wadliwe cytaty lub struktura,
niepuste odpowiedzi bez twierdzeń klinicznych i awarie weryfikatora. Dodatkowo
kontrolowane jest losowe 10% list faktów i 10% notatek osobno w E/A/B/C,
z zaokrągleniem w górę i minimum jedną pozycją na niepustą grupę. Losowanie
nie zależy od etykiet; ziarno wynosi 27182. Nie ogranicza się liczby dodatkowych
przypadków wymagających kontroli.

Każdy pakiet może zatwierdzić student_1 albo student_2 po porównaniu ze źródłem.
Wątpliwości pozostają otwarte do wspólnego rozstrzygnięcia. Trzecia rola nie jest
wymagana. Pakiety nie pokazują wariantu ani ziarna. Kontroler sprawdza również
pominięte fakty i twierdzenia. Eksport wymaga zamknięcia całej kolejki.

Zapisy zachowują surowe odpowiedzi, pochodzenie ocen, osobę kontrolującą,
uzasadnienie i sumy kontrolne wejść. Raport audytu podaje liczebność próbki
oraz liczbę korekt. Nie raportuje się zgodności ludzi na podstawie wywołań modelu.
Generator i weryfikator należą do tej samej rodziny; ryzyko wspólnych błędów
ogranicza interpretację wyników. Szczegóły zawiera
[opis oceny automatycznej](AUTOMATIC_EVALUATION.md).

## Metryki i braki danych

Dla każdej notatki:

- `T = P + S + N`.
- `H = (S + N) / T` dla `T > 0`; w przeciwnym razie `H` jest nieokreślone (`null`).
- `H* = H` dla odpowiedzi z twierdzeniami klinicznymi.
- `H* = 1` dla pustej odpowiedzi, całkowitej odmowy lub braku twierdzeń klinicznych.
- `C = liczba poprawnie odtworzonych faktów / K`.
- `O = 1 - C`.

Odpowiedzi bez twierdzeń mają zero odtworzonych faktów. Odmowa zawierająca również
twierdzenia kliniczne jest oceniana na podstawie tych twierdzeń.

Najpierw uśrednia się trzy powtórzenia danej konsultacji i wariantu, następnie
oblicza makrośrednią po konsultacjach. Główną miarą wierności jest `H*`.
`H` uwzględnia tylko określone wartości; należy podać jego mianownik i nie
zastępować nim `H*`. Mikrośrednia po wszystkich twierdzeniach nie jest główną
miarą. E ma jedną deterministyczną obserwację.

Awaria techniczna nie jest błędną odpowiedzią kliniczną. Wszystkie zadania,
także nieudane, pozostają w danych. Gdy braki techniczne przekroczą 5% zadań
dowolnego wariantu, analiza główna zostaje zatrzymana. Przy brakach do 5% średnie
opisowe obejmują kompletne grupy trzech powtórzeń, a cztery testy — wspólny zbiór
kompletnych konsultacji A/B/C. Niepełne grupy i liczebność próby sparowanej muszą
być jawnie podane. Ta polityka wymaga potwierdzenia przed eksperymentem.
Słaba treść odpowiedzi nie jest podstawą usunięcia konsultacji. Wyników
z różnych środowisk obliczeniowych nie wolno łączyć.

## Zaplanowana analiza statystyczna

Na `test1`, po uśrednieniu powtórzeń w konsultacjach, sprawdzane są cztery hipotezy. W bieżącej wersji dotyczą one wskaźników weryfikatora i mają charakter eksploracyjny:

1. `H*_C - H*_A < 0`.
2. `H*_C - H*_B < 0`.
3. `C_C - C_A > -0.05`.
4. `C_C - C_B > -0.05`.

Pierwsze dwie wykorzystują `scipy.stats.permutation_test` dla średniej różnicy
sparowanej: `permutation_type="samples"`, `alternative="less"`, 99999 resamplingów.
Jeśli istnieje mniej różnych permutacji, SciPy oblicza rozkład dokładny.
Dwie pozostałe korzystają z
`ttest_1samp(paired_differences, popmean=-0.05, alternative="greater")`.
Wszystkie cztery wartości p są korygowane łącznie:
`multipletests(method="holm", alpha=0.05)`.

Nie wolno zastępować analizy testem Friedmana ani wybierać testu po obejrzeniu
wyników. Zerowa wariancja różnic kompletności blokuje test t; nie uruchamia
automatycznie innej procedury. Istotność statystyczna sama nie przesądza
o znaczeniu klinicznym.

Przedziały 95% wyznacza sparowany bootstrap percentylowy z 10000 losowań całych
konsultacji, z zachowaniem sparowania metod i średnich po ziarnach. Są to
dwustronne przedziały opisowe, bez korekty wielokrotnych porównań.
E nie uczestniczy w czterech testach.

Zgodność surowa, kappa Cohena i macierz P/S/N są dostępne wyłącznie w opcjonalnym trybie pełnej, niezależnej anotacji ręcznej. Wersja modelowa raportuje kontrole i korekty w próbce audytowej, bez kappy między wywołaniami modelu.

## Decyzje wymagające potwierdzenia przed zamrożeniem

- Oficjalne pełne pliki CSV jako podstawowa reprezentacja transkrypcji.
- Przegląd 60 reguł bez wglądu w wyniki testowe.
- Model weryfikujący, odsetek audytu, wymagane dowody i podział kontroli między dwie osoby.
- Wymóg `K > 0`, obsługa braków technicznych, konwencja przedziałów bootstrapowych
  i zatrzymanie przy zdegenerowanym teście t.
- Sprzęt, ewentualny wyjątek kwantyzacji oraz niezmienne wersje przebiegu i kodu.

## Źródła techniczne

- [Oficjalne dane ACI-Bench](https://github.com/wyim/aci-bench).
- [Generator Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- [API enkodera](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
- [Sparowany test permutacyjny](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
- [Korekta Holma](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
