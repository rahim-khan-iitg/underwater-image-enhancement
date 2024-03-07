### Step 1 : Color Correction:-
for each channel in RGB image do the color correction according to the given formula :-
```math
U^c=\frac{255}{2} \left(1+\frac{S^c-M^c}{\mu V^c} \right)
```
where
$`S^c`$ = Color channel 
$ `M^c`$ = Mean of the channel
$`V^c`$ = Varience of the channel
