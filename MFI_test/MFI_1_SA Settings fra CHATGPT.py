SA Settings fra CHATGPT

INITIAL_TEMPERATURE
# Bestemmer hvor villig SA er til at acceptere dårligere løsninger i starten.
# Høj temperatur -> meget exploration, accepterer mange dårlige løsninger, mere random søgning
# Lav temperatur -> mere greedy, hurtigere convergence, større risiko for local optimum

COOLING_RATE
# Hvor hurtigt temperaturen falder.
# T_new = T_old * COOLING_RATE
# Så cooling rate bestemmer temperatur fladet pr. iteration
# Høj cooling rate -> langsom cooling, længere exploration, bedre løsninger, længere runtime
# Lav cooling rate -> hurtig cooling, hurtigere convergence, mere greedy behavior

MIN_TEMPERATURE
# Stopkriteriet - når temperaturen falder hertil stopper simuleringen
# MIN_TEMPERATURE = 0.1 -> når T < MIN_TEMPERATURE så stopper SA
# Man kan også indføre en maks. iteration som stopkriterie (men det er allerede hvad min. temperatur er)

#De her tre bestemmer i hvilken grad SA laver hvilke operationer -> skal tilsammen give 1.0
SA_RANDOM_INSERT_RATE
# Flyt én ordrer tilfældigt f.eks. [1,2,3,4] -> fyltter 4 [1,4,2,3]
SA_LATE_ORDER_INSERT_RATE
# Her flyttes en af de forsinkede ordrer frem i rækkefølgen f.eks. [1,2,3,4] 
# -> 4 er forsinket og flyttes frem -> [4,1,2,3]
SA_SWAP_RATE
# Her byttes to ordrer f.eks. [1,2,3,4] -> 1 og 4 bytter -> [4,2,3,1]

#Færre iterationer ændr cooling rate
#| Ønskede iterationer | Cooling rate |
#| ------------------- | ------------ |
#| 25                  | 0.759        |
#| 50                  | 0.871        |
#| 100                 | 0.933        |

# Standard de her værdier
SA_INITIAL_TEMPERATURE = 100
SA_MIN_TEMPERATURE = 0.1

SA_RANDOM_INSERT_RATE = 0.3
SA_LATE_ORDER_INSERT_RATE = 0.5
SA_SWAP_RATE = 0.2


#CHATGPT for 1 dag (ca. 273 iterationer)
SA_COOLING_RATE = 0.975

#CHATGPT for 3 dage (ca. 459 iterationer)
SA_COOLING_RATE = 0.985

#Chatgpt vil starte med denne (ca. 342 iterationer)
SA_COOLING_RATE = 0.98

#Overvej at køre en runde med disse indstillinger
SA_RANDOM_INSERT_RATE = [0.2-0.8]
SA_LATE_ORDER_INSERT_RATE = [0.2-0.8]
SA_SWAP_RATE = [0.2]
