# Thatcher-effect
```
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```
-   az algoritmus szükeárnyalatos képet vár
-   scalefactor = kell egy kiinduló méret, savlefactorszorosára csökkentjük minden iterációban 
nagyobb -> gyorsabban fut, pontatlanabb
kisebb -> fordíva
-   minNeighbours = legalább ennyi háromszöget kell találnia hogy arcnak kategorizálja
-   minsize = háromszög minimum mérete
-   maxsize = háromszög maximum mérete
