GA Settings fra CHATGPT

GA Settings til 1 dag fra CHATGPT

SWAPS             
# Bruges kun til initial population creation ud fra initial chromosome f.eks. et swap [1,2,3] -> [3,2,1]
# Swaps skal nok ikke være for højt hvis vi forventer at EDD er en god begyndelse!!!

POPULATION_SIZE     
# Er antal chromosomer/individer der laves i en generation f.eks. [1,2,3] = 1 chromosom/individ
# Flere chromosomer giver en større undersøgelse i hver generation

GENERATIONS         
# Population i én generation f.eks. kan [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] være en population i en generation /iteration
# Flere generationer -> flere iterationer groft sagt

ELITE_SIZE            
# Diktere hvilke chromosomer/individer der overlever fra en generation til en anden urørt 
# f.eks. hvis elite size = 1, generation 1 bedste chromosom [1,2,3] bliver ført videre til generation 2 som [1,2,3]
# Hvor meget af en ny generation skal afhænge af den tidligere generation - for lav risikere at miste gode løsninger - for højt kan fange os i et lokalt minimum

TOURNAMENT_SIZE        
# Bestemmer hvem der udvælges som parent i en generation (inkkl. elites) 
# f.eks. chromosomer a,b,c,d,e,f,g,h (med bedste fitness først) -> tournament på 3 
# -> 3 kan vælges som parent 1 [a,c,h] -> a har bedste fitness -> 2 kan vælges som parent 2 [b,c,d] 
# -> b har bedste fitness -> a og b er parents til child 1 -> kør igen

CROSSOVER_RATE   
# Bestemmer chancen for at en child er en blanding af parents 
# f.eks. 80% så er der 80% chance for at child er en blanding af parents 
# og så 20% chance for at child er en kopi af den ene parent
# Høj crossover rate undersøger flere muligheder
# Lav crossover rate udnytter allerede gode løsninger / sikrer stabilitet

MUTATION_RATE 
# Bestemmer chancen for at der sker en mutation i en child
# Mutationen betyder at man tager en vilkårlig ordrer ud og placere den et andet sted
# F.eks. [1,2,3] -> 1 tages ud -> [2,3,1]
# Forskel på swaps er at den bytter to ordrer / mutation flytter én ordrer
# en mutation sikrer at den søger rundt og ikke sidder fast i de samme løsninger (da disse er kombinationer fra deres parents)

#Udfra et relativt godt startbud er min intiution at vi lander i settings omkring 1-3 dage ->5 dage er for voldsomt da et stor del af løsningsrummet ender med at blive dårligt
#Mit umiddelbare bud

#Mit bud
SWAPS = 2
POPULATION_SIZE = 10
GENERATIONS = 20
ELITE_SIZE = 1
TOURNAMENT_SIZE = 2
CROSSOVER_RATE = 0.90 
MUTATION_RATE = 0.15  

#Chattens bud til 1 dag
SWAPS = 2-3
POPULATION_SIZE = 8-15
GENERATIONS = 15-40
ELITE_SIZE = 1
TOURNAMENT_SIZE = 2
CROSSOVER_RATE = 0.85-0.95
MUTATION_RATE = 0.10-0.25

#Chattens bud til 3 dage
SWAPS = 3-6
POPULATION_SIZE = 25-50
GENERATIONS = 80-200
ELITE_SIZE = 2-4
TOURNAMENT_SIZE = 3-4
CROSSOVER_RATE = 0.85-0.95
MUTATION_RATE = 0.15-0.30

GA Settings chattens bud til en start (v. 1-3 dage)

SWAPS = 4
POPULATION_SIZE = 35
GENERATIONS = 120
ELITE_SIZE = 3
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2

#| Horizon | Orders ca. | Population | Generations |
#| ------- | ---------: | ---------: | ----------: |
#| 1 dag   |        4-5 |      10-20 |       20-50 |
#| 3 dage  |         13 |      30-60 |     100-300 |
#| 5 dage  |         22 |     60-120 |     200-600 |