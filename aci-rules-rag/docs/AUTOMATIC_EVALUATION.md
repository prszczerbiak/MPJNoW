# Ocena modelowa z audytem

## Zakres i status metodologiczny

Tryb `model_assisted_v2` zastępuje pełną ręczną anotację w podstawowym przebiegu
projektu studenckiego. Jest zmianą względem pierwotnego protokołu opartego
na dwóch niezależnych anotatorach i trzeciej osobie rozstrzygającej spory.
Nie odtwarza tej procedury ani nie może być raportowany jako jej wykonanie.

Zachowano wzory metryk i plan porównań A/B/C. Zmieniło się źródło etykiet,
podziału na twierdzenia i list wymaganych faktów. H, H*, C i O są zatem
wskaźnikami oceny modelowej z częściową kontrolą człowieka. Analiza ma charakter
eksploracyjny, a jej wnioski zależą od jakości weryfikatora.

## Weryfikator i powtarzalność

Domyślny model to `Qwen/Qwen2.5-1.5B-Instruct`. To osobny zestaw wag względem
generatora 3B, lecz oba modele należą do tej samej rodziny. Ich błędy mogą być
skorelowane. Użycie dokładnie modelu generatora jako weryfikatora jest blokowane.

Konfiguracja `configs/evaluation.yaml` zawiera rewizję modelu, precyzję,
urządzenie, budżet kontekstu, limit odpowiedzi, ziarno i odsetek audytu.
Weryfikator korzysta z 4-bitowej kwantyzacji NF4, podwójnej kwantyzacji i obliczeń bfloat16.
Wymaga CUDA i jest ładowany po zwolnieniu generatora. Generowanie oceny jest zachłanne (`do_sample=False`).
Domyślny budżet 16384 tokenów obejmuje wejście oraz 6144 tokeny odpowiedzi.
Przekroczenie limitu kieruje sprawę do kontroli zamiast przycinać transkrypcję.

`configure_run` automatycznie uzupełnia brakującą rewizję. Możliwe jest także
osobne polecenie `python -m src.evaluate_notes pin`. Rewizji nie można zmieniać
po utrwaleniu manifestu oceny.

Manifest wiąże oceny z danymi, odpowiedziami generatora, konfiguracją,
promptami, wersjami bibliotek, GPU i kodem. Wywołania są zapisywane przed
wykonaniem, a odpowiedzi surowe są zachowywane. Wznowienie nie powtarza ocen.
Przerwane lub wadliwe wywołanie kieruje sprawę do ręcznej kontroli; nie jest
ponawiane aż do uzyskania korzystniejszej etykiety.

## Listy faktów

Etap `facts` otrzymuje wyłącznie transkrypcję. Nie ma dostępu do notatek
wygenerowanych, referencji, wariantu ani ziarna generatora. Lista musi zawierać
unikalne identyfikatory, kategorie, treść, dokładne fragmenty źródła
i uzasadnienia istotności. Pusta lista wymaga ręcznej korekty.

Automatyczna kontrola cytatu potwierdza obecność tekstu, nie poprawność
interpretacji ani wybór wszystkich istotnych faktów. Dlatego przypadki
oznaczone jako niepewne oraz losowa próbka list wymagają przeglądu.
Żadna notatka nie jest oceniana przed rozstrzygnięciem kolejki faktów.

## Dwa przebiegi oceny notatki

Pierwszy przebieg tworzy pełny podział na twierdzenia, etykiety P/S/N i ocenę
odtworzenia każdego faktu. Drugi otrzymuje tę samą transkrypcję, notatkę
i ustalony podział twierdzeń, lecz nie etykiety ani uzasadnienia pierwszej oceny.
Kolejność faktów jest odwrócona. Zadaniem drugiego przebiegu jest także
zasygnalizowanie pominięć i błędów podziału.

Porównanie dotyczy etykiet, odtworzenia faktów i powiązań z twierdzeniami.
Różnice stylistyczne uzasadnień nie są uznawane za rozbieżność.
Oba przebiegi korzystają z tego samego modelu i nie stanowią niezależnych ocen
ludzkich. Zgodność może wynikać ze wspólnego błędu.

Wymagania walidatora:

- Każde twierdzenie ma identyfikator i dosłowny fragment notatki.
- P i S wskazują dosłowny fragment transkrypcji.
- Każde twierdzenie i wymagany fakt są ocenione dokładnie raz.
- Poprawnie odtworzony fakt wskazuje co najmniej jedno twierdzenie P.
- Odpowiedzi bez twierdzeń nie mogą mieć odtworzonych faktów.
- Pusta odpowiedź jest rozpoznawana deterministycznie i ma H*=1.
- Odmowy i brak treści klinicznej w niepustym tekście wymagają kontroli człowieka.

Pole `uncertain` sygnalizuje potrzebę przeglądu. Nie jest skalibrowanym
prawdopodobieństwem ani miarą pewności klinicznej.

## Kolejka kontroli

Do kolejki trafiają błędy formatu i infrastruktury, nieistniejące cytaty,
rozbieżności, niepewność i przypadki z próbki losowej.
Losowanie jest deterministyczne, niezależne od etykiet:
10% konsultacji dla list faktów i 10% notatek osobno dla E/A/B/C, z zaokrągleniem
w górę i minimum jedną pozycją w niepustej grupie. Wariant i ziarno nie są
pokazywane w pakiecie ani przekazywane weryfikatorowi.

Przy 40 konsultacjach, bez wykluczeń i braków technicznych, próbka obejmie
4 listy faktów i 40 notatek: 4 z E oraz po 12 z A/B/C.
Pozostałe skierowania do kontroli zależą od faktycznej jakości ocen.
Nie ma limitu pozwalającego pominąć trudne przypadki.

Kontroler korzysta z polecenia:

```bash
python -m src.evaluate_notes review --reviewer student_1
```

Możliwe decyzje:

| Opcja | Działanie |
|---|---|
| `a` | Zatwierdzenie propozycji po sprawdzeniu źródła; usuwa znaczniki niepewności |
| `e` | Wczytanie poprawionego obiektu JSON o tym samym schemacie |
| `q` | Zakończenie sesji bez zatwierdzania pozostałych pozycji |

Zatwierdzenie wymaga uzasadnienia po angielsku. Przy kontroli należy sprawdzić
również brakujące fakty i twierdzenia, nie tylko proponowane etykiety.
Wątpliwość, której jedna osoba nie potrafi rozstrzygnąć, pozostaje otwarta
do konsultacji z drugą osobą. Nie jest wymagana trzecia osoba.
Ocena modelu nie stanowi dowodu przy rozstrzyganiu sporu.

Podział pracy odbywa się przez kolejne sesje `student_1` i `student_2`.
To narzędzie plikowe bez uwierzytelniania; zapis równoległy nie jest obsługiwany.
Koordynatorzy nie powinni wykorzystywać prywatnego mapowania do wyboru etykiet.

## Pliki i eksport

| Ścieżka w katalogu oceny | Zawartość |
|---|---|
| `manifest.json` | Utrwalone wejścia, konfiguracja, prompty i środowisko |
| `calls/` | Surowe odpowiedzi weryfikatora i stan wywołań |
| `facts/` | Przyjęte listy faktów z pochodzeniem |
| `queue/` | Pakiety kontroli; pozostają dostępne po zatwierdzeniu |
| `decisions/` | Przyjęte oceny automatyczne i decyzje ludzi |
| `private/mapping.json` | Powiązanie identyfikatorów kontroli z zadaniami generatora |
| `metrics.csv` | Metryki z oznaczeniem metody, środowiska i pochodzenia oceny |
| `summary.json` | Liczby ocen automatycznych, kontroli i korekt w losowej próbce |

Pole `assessment_origin` przyjmuje `automatic` albo `human_review`.
Ocena modelowa z ręczną korektą nadal ma `evaluation_method=model_assisted_v2`,
ponieważ nie jest pełną, niezależną anotacją całego zbioru.

Eksport blokuje się przy nierozstrzygniętych kontrolach. Brak oceny nie
powoduje usunięcia trudnej notatki ani przypisania jej korzystnego wyniku.
Awarie generatora zachowują status `technical_failure` i nie otrzymują
oceny klinicznej.

`summary.json` podaje liczbę skorygowanych pozycji w losowej próbce.
Za korektę uznaje zmianę listy faktów, twierdzeń, etykiet, powiązań odtworzenia
lub rodzaju odpowiedzi, nie samą redakcję uzasadnienia.
Nie jest to estymator błędu na poziomie pojedynczego twierdzenia.
Wysoki udział korekt wymaga ograniczenia wniosków i przeglądu metody na train/valid;
nie uzasadnia dostrajania weryfikatora do test1.

## Analiza i ograniczenia

`aggregate_results` wymaga jawnego `--evaluation-mode model-assisted`.
Nie dopuszcza mieszania ocen ręcznych i modelowych ani różnych wersji weryfikatora.
Raport statystyczny zawiera `inference_scope=exploratory_model_assisted`.
Nie oblicza się kappy Cohena między wywołaniami modelu.

Przegląd literatury wskazuje m.in. obciążenia związane z długością odpowiedzi,
kolejnością i preferowaniem własnych odpowiedzi przez modele:
[Zheng i in., 2023](https://arxiv.org/abs/2306.05685).
Publikacja nie potwierdza poprawności klinicznej wybranego modelu 1.5B.
Jego jakość w tym zadaniu pozostaje do sprawdzenia; testy oprogramowania
weryfikują działanie potoku, a nie trafność ocen medycznych.
