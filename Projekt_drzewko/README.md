| 1 class 

affordable:    unacc, acc, good, vgood

| 6 attributes 

buying:   vhigh, high, med, low. (Jaka jest cena zakupu)

maint:    vhigh, high, med, low. (Jaki jest koszt utrzymania)

doors:    2, 3, 4, 5more. (Ilość drzwi w samochodzie)

persons:  2, 4, more. (Ilość osób do przewozu w samochodzie)

lug_boot: small, med, big. (Wielkość bagażnika)

safety:   low, med, high. (Bezpieczeństwo samochodu)
| Ilość rekordów 1728
| Brak missing_values


Dokonujemy Klasyfikacji opłacalności kupna samochodu na podstawie kolumny affordable 
Są wartości spoza akceptowalnych wartosci i sa zmienione na poprzednia wartosc z rekordu wyzej poniewaz cala baza jest posortowana.
Nie usuwalem tych wartosci i tak wykryto bledy.

ROZSZERZONA KLASYFIKACJA I ANALIZA
W Bazie nie pojawiły się poważniejsze błędy oprócz innych wartości typu string oraz Nan i ujemnych wartości lub o złym typie.
Drzewko Klasyfikacji poradziło sobie najlepiej ze wszytkich.
Im większy zbiór treningowy tym lepsze wyniki ale o niewiele. (średnio 3 procent)