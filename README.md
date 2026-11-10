KLASY WYBRANE:
1 background 1     !
2 facade 2         !
3 window 10        !
4 door 5           !
________________________________
5 cornice 11       <- do fasady
________________________________
6 sill 3           <- do okna
________________________________
7 balcony 4        !
________________________________
8 blind 6          <- do okna
________________________________
9 deco 8           <- do fasady
________________________________
10 molding 7       <- do fasady
________________________________
11 pillar 12       !
________________________________
12 shop 9          <- do drzwi

WYMAGANIA FORMALNIE DATASET:
1. Żeby to było jakoś sensownie w ndarrayach
2. Robimy tylko maski, jajca w xmle
3. Jakieś metody get_train, val, test i żeby to co dostanę było reprodukowalne (czyli wszędzie seedzik)
4. Wspólny preprocessing:
- napewno jakaś jedna rozdzielczość, tworzący dataset sam dobiera rozmiar (Wymysl jak)
- sprowadzenie to powyższych klas (może zachować jakąś metodę get og bez tych naadpisań jęsli to nie były problem?)