# Instrukcja ręcznej anotacji

Obowiązują dwie niezależne oceny i trzecia osoba rozstrzygająca.
Jest to podstawowa i jedyna procedura oceny treści w projekcie.

Oceny kliniczne należą do anotatorów. Sprawdzenie formatu, identyfikatora lub
obecności cytatu nie jest oceną prawdziwości twierdzenia. Etykiet nie należy
zastępować werdyktem modelu.

Koordynator przechowuje klucze mapowania. Każdy anotator otrzymuje tylko własny
katalog `packets/annotator_N`. Obsługa plików nie zapewnia uwierzytelniania
ani izolacji użytkowników; niezależność pracy wymaga oddzielnych kont lub
kontrolowanego udostępniania plików.

## 1. Wymagane fakty przed udostępnieniem notatek

Uruchom `python -m src.prepare_annotation facts`. Pakiety zawierają transkrypcje,
bez notatek wygenerowanych i referencyjnych. Każdy anotator niezależnie tworzy
listę obejmującą powód wizyty, istotne objawy i negacje, leki i zmiany leczenia,
wyniki i badania, rozpoznania z poziomem pewności oraz plan.

Uwzględniaj tylko istotne fakty obecne w transkrypcji. Brak wzmianki nie oznacza
prawidłowego wyniku ani nieobecności choroby.

Przykład syntetycznego zgłoszenia; treść danych pozostaje po angielsku:

```json
{
  "encounter_id": "synthetic-0",
  "annotator_id": "annotator_1",
  "facts": [{
    "fact_id": "F001",
    "category": "symptoms",
    "text": "The patient reports a cough.",
    "source_span": "I have a cough.",
    "rationale": "Presenting symptom relevant to the visit."
  }]
}
```

`source_span` jest dosłownym fragmentem transkrypcji. `text` może być krótką,
wierną parafrazą. Dostępne kategorie: `visit_reason`, `symptoms`, `negations`,
`medications`, `results_tests`, `diagnoses_uncertainty`, `plan`.
Identyfikatory faktów muszą być unikalne w konsultacji.

Koordynator importuje oba niezależne pliki:

```bash
python -m src.annotate submit-facts path/to/annotator_1_facts.json
python -m src.annotate submit-facts path/to/annotator_2_facts.json
```

Trzeci anotator przygotowuje plik uzgodnienia z polami `encounter_id`,
`adjudicator_id: "annotator_3"`, `rationale` i końcową listą `facts`.
Dla każdej konsultacji uruchom:

```bash
python -m src.annotate freeze-facts path/to/consensus.json
```

Oryginalne zgłoszenia pozostają niezmienione. List faktów nie można modyfikować
po udostępnieniu notatek.

## 2. Zaślepienie notatek i podział na twierdzenia

Po zakończeniu generowania i zamrożeniu wszystkich list faktów:

```bash
python -m src.prepare_annotation notes
```

Koordynator otrzymuje `segmentation/N*.json` i prywatny klucz mapowania.
Kolejność notatek jest losowana z ziarnem `27182`. Anotatorom nie udostępnia się
kluczy, nazw wariantów, ziaren, promptów, dobranych reguł, logów generowania
ani metadanych technicznych.

Anotatorzy sporządzają pełną listę twierdzeń atomowych. Zdanie opisujące dwa
niezależne objawy daje dwa twierdzenia. Nazwa leku, dawka, droga i częstość podania
powinny dać się zweryfikować osobno. Twierdzenia niepotwierdzone również należą
do listy — ich usunięcie zaniżałoby mianownik metryki.

Nagłówki i sam znacznik `Not mentioned in the transcript` nie są twierdzeniami
klinicznymi. Nie licz tego samego twierdzenia dwukrotnie, ale zachowaj rozróżnienia
dotyczące czasu i osoby, której dotyczy informacja.

Obaj główni anotatorzy przeglądają podział, nie uzgadniając przy tym etykiet
prawdziwości. Pominięte lub nadmiernie połączone twierdzenia należy poprawić
przed rozpoczęciem niezależnej oceny. Plik podziału ma postać:

```json
{
  "blind_id": "N_FROM_PACKET",
  "reviewed_by": ["annotator_1", "annotator_2"],
  "response_kind": "clinical",
  "claims": [{"claim_id": "C001", "text": "The patient reports a cough.",
              "note_span": "The patient has a cough."}]
}
```

Zatwierdź podział:

```bash
python -m src.annotate freeze-claims path/to/claims.json
```

Polecenie tworzy osobne pakiety dla obu anotatorów. Dla pustej odpowiedzi, całkowitej
odmowy lub braku treści klinicznej użyj odpowiednio `empty`, `refusal` albo
`no_clinical_claims` z pustą listą twierdzeń. Klasyfikację potwierdzają ludzie.
Odmowa zawierająca twierdzenia kliniczne otrzymuje `clinical`.

## 3. Niezależne etykiety i odtworzenie faktów

- **P:** twierdzenie potwierdzone w źródle, zgodne co do osoby, czasu, pewności
  oraz obecności lub braku zjawiska.
- **S:** twierdzenie sprzeczne ze źródłem.
- **N:** twierdzenie niepotwierdzone, ale niesprzeczne. Sam brak wzmianki
  nie stanowi sprzeczności.

S i N łącznie tworzą kategorię niepotwierdzonych w metryce H. Każda etykieta
wymaga uzasadnienia, a P i S — dosłownego fragmentu potwierdzającego lub przeczącego.
Dla N opisz brak potwierdzenia; nie twórz fikcyjnego cytatu.

Dla każdego wymaganego faktu określ, czy odtworzono go poprawnie, i wskaż
potwierdzone twierdzenia P. Błędne przywołanie faktu nie liczy się jako poprawne
odtworzenie.

```json
{
  "blind_id": "N_FROM_PACKET",
  "annotator_id": "annotator_1",
  "labels": [{"claim_id": "C001", "label": "P", "source_span": "I have a cough.",
              "rationale": "Explicit patient report."}],
  "coverage": [{"fact_id": "F001", "covered": true, "claim_ids": ["C001"],
                "rationale": "The required symptom is reproduced accurately."}]
}
```

Import oceny:

```bash
python -m src.annotate submit-rating path/to/rating.json
```

Obie niezależne oceny muszą istnieć przed rozstrzyganiem sporów. Nieodtworzony fakt
otrzymuje `covered: false`. Odpowiedź bez twierdzeń ma `labels: []` i wszystkie
fakty oznaczone jako nieodtworzone.

Po zakończeniu pracy obu anotatorów oblicz zbiorczą zgodność sprzed uzgodnienia:

```bash
python -m src.annotate agreement outputs/agreement.json
```

## 4. Rozstrzyganie sporów

Trzeci anotator sprawdza różnice w etykietach i ocenach odtworzenia faktów,
zapisuje uzasadnienie i zatwierdza końcowe tablice `labels` oraz `coverage`.
Stosuje ten sam format, lecz zamiast `annotator_id` podaje
`adjudicator_id: "annotator_3"` i dodaje główne pole `rationale`.

```bash
python -m src.annotate adjudicate path/to/decision.json
```

Pierwotne oceny, zgodność sprzed uzgodnienia i decyzje końcowe są przechowywane
oddzielnie. Sumy kontrolne wskazują użyte wersje list faktów i zgłoszeń.
Program nie wybiera etykiety końcowej. Metryki korzystają wyłącznie
z katalogu `adjudicated`.

Ponowny import identycznej treści jest bezpieczny; nadpisanie inną treścią jest
odrzucane. Błędne zgłoszenie wymaga udokumentowanej nowej wersji zestawu anotacji
z zachowaniem oryginału. Nie edytuj zatwierdzonych plików w miejscu.
