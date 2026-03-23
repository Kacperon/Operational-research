# Opis Problemu i Funkcja Kosztu

Poniższy dokument przedstawia matematyczne sformułowanie problemu optymalizacji planu treningowego. Celem jest znalezienie takiej sekwencji ćwiczeń, która w sposób zrównoważony obciąża grupy mięśniowe, uwzględniając jednocześnie specyficzne preferencje użytkownika.

## Dane

W modelu wykorzystujemy następujące parametry:

* **$G$ — Grupy mięśniowe:** Zbiór wszystkich grup mięśniowych branych pod uwagę.
* **$T \subseteq G^3$ — Dostępne ćwiczenia:** Zbiór dostępnych ćwiczeń, gdzie każde ćwiczenie jest zdefiniowane jako trójka: (**target** - docelowa, **synergist** - współpracująca, **stabilizer** - stabilizująca).
* **$n \in \mathbb{N}$ — Liczba ćwiczeń:** Planowana liczba ćwiczeń do wykonania w danej jednostce treningowej.
* **$\alpha \in \mathbb{R}^{|G|}$ — Wagi grup mięśniowych:** Wektor wag określający priorytety dla poszczególnych grup mięśniowych (wyższa waga = większy priorytet).
* **$\beta \in \mathbb{R}^3$ — Wagi stopni intensywności:** Wektor wag przypisany do ról mięśni (target, synergist, stabilizer), określający siłę ich zaangażowania.
* **$\gamma \in \mathbb{R}$ — Współczynnik równowagi:** Parametr kontrolujący, jak ważna jest dla nas równomierność treningu w stosunku do preferencji $\alpha$.

---

## Szukane

Celem jest znalezienie sekwencji $n$ ćwiczeń $x \in T^n$, która minimalizuje funkcję kosztu:

$$C(X) = \gamma \left\| \overline{\beta^T X} - \beta^T X \right\|_2 - \beta^T X \alpha$$

Gdzie macierz $X$ (o wymiarach $3 \times |G|$) jest reprezentacją planu $x$ taką, że:
* **$X_{ij}$** to liczba wystąpień $j$-tej grupy mięśniowej w $i$-tym stopniu intensywności (roli) w całym planie $x$.

### Wyjaśnienie składników funkcji kosztu:

1.  **$\beta^T X$**: Wektor reprezentujący całkowite, ważone obciążenie dla każdej grupy mięśniowej.
2.  **$\overline{\beta^T X}$**: Wektor średniego obciążenia (idealny stan, w którym każdy mięsień pracuje tyle samo).
3.  **$\gamma \left\| \dots \right\|_2$**: Składnik odpowiedzialny za **wariancję**. Minimalizacja tego członu dąży do tego, aby różnice w obciążeniu poszczególnych mięśni były jak najmniejsze (równomierny rozwój).
4.  **$-\beta^T X \alpha$**: Składnik odpowiedzialny za **preferencje**. Minimalizacja tego członu (poprzez odejmowanie) promuje wybieranie tych grup mięśniowych, które mają przypisane wysokie wagi w wektorze $\alpha$.

---

## Ogólny pomysł

Logika optymalizacji opiera się na dwóch filarach:
* **Równowaga ($\gamma$):** Chcemy uniknąć sytuacji, w której niektóre grupy mięśniowe są skrajnie przetrenowane, a inne pominięte.
* **Personalizacja ($\alpha$):** Jeśli mamy kilka podobnych rozwiązań o zbliżonej równowadze, wybieramy to, które kładzie większy nacisk na grupy mięśniowe, na których najbardziej nam zależy.
