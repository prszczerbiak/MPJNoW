# Protokół badawczy — wersja 4

Repozytorium zawiera środowisko do przeprowadzenia badania. Artefakty przebiegów
są zapisywane lokalnie w `outputs/`.
Analiza główna obejmuje oficjalny zbiór `test1`. Zbiory `train`
i `valid` służą do kontroli technicznej i dopracowania protokołu przed jego
zamrożeniem. `test2` i `test3` są przeznaczone do osobno zadeklarowanych analiz
dodatkowych. Nie wolno dobierać parametrów na zbiorach testowych, łączyć podziałów
ani stosować walidacji krzyżowej.

## Wersja metody oceny

Obowiązuje pełna anotacja ręczna. Dwie osoby niezależnie przygotowują fakty
oraz oceniają notatki; trzecia osoba uzgadnia listy i rozstrzyga oceny.
Program nie tworzy list faktów ani etykiet P/S/N. Formularze, walidacja,
zaślepienie, obliczanie zgodności i metryk pozostają zautomatyzowane.
Metryki są oznaczone jako `evaluation_method=human`.

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

Do ładowania generatora służy bitsandbytes. Profil wymaga PyTorch z CUDA i jawnie wybiera cuda:0;
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

## Ręczna ocena

Przed udostępnieniem notatek dwaj anotatorzy niezależnie tworzą wymagane fakty
z transkrypcji. Każdy fakt zawiera kategorię, treść, dokładny cytat i uzasadnienie
istotności. Trzeci anotator zatwierdza końcową listę. K musi być dodatnie.
Po udostępnieniu notatek listy faktów pozostają zamrożone.

Notatki są zaślepiane: pakiety nie ujawniają wariantu, ziarna ani promptu.
Obaj anotatorzy sprawdzają ręczny podział na twierdzenia, po czym niezależnie
nadają P/S/N oraz oceniają odtworzenie każdego faktu. P i S wymagają cytatu.
Człowiek klasyfikuje także puste odpowiedzi, odmowy i brak twierdzeń klinicznych.

Trzeci anotator zapisuje końcowe decyzje i uzasadnienia. Program zachowuje
zgłoszenia pierwotne oraz ich sumy kontrolne i nie wybiera rozstrzygnięcia.
Zgodność surową, kappę Cohena i macierz P/S/N oblicza się przed uzgodnieniem.
Pełne formaty i kolejność pracy opisuje `ANNOTATION_GUIDELINES.md`.

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

Na `test1`, po uśrednieniu powtórzeń w konsultacjach, sprawdzane są cztery hipotezy. Metryki opierają się na ręcznie zatwierdzonych ocenach:

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

Zgodność surowa, kappa Cohena i macierz P/S/N opisują niezależne oceny ludzi sprzed uzgodnienia.

## Decyzje wymagające potwierdzenia przed zamrożeniem

- Oficjalne pełne pliki CSV jako podstawowa reprezentacja transkrypcji.
- Przegląd 60 reguł bez wglądu w wyniki testowe.
- Przydział dwóch niezależnych anotatorów i trzeciej osoby rozstrzygającej oraz wymagane dowody źródłowe.
- Wymóg `K > 0`, obsługa braków technicznych, konwencja przedziałów bootstrapowych
  i zatrzymanie przy zdegenerowanym teście t.
- Sprzęt, ewentualny wyjątek kwantyzacji oraz niezmienne wersje przebiegu i kodu.

## Źródła techniczne

- [Oficjalne dane ACI-Bench](https://github.com/wyim/aci-bench).
- [Generator Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- [API enkodera](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
- [Sparowany test permutacyjny](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
- [Korekta Holma](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
