import matplotlib.pyplot as plt

fitness = [97.46114246772281, 97.14348173541663, 117.59278458471563, 112.50447918296666, 124.06595900576536, 127.09653248862226, 140.82979331119807, 125.83012535346701, 140.9804569606844, 147.90814214280527, 157.4553193902952, 137.196371578406, 148.25827440968146, 164.59357343601607, 137.5491452383391, 144.89970084450127, 165.83281459945337, 166.76516185643376, 167.94117647058823, 155.28458410440035, 155.90628116365977, 153.57416365810445, 162.30893400804575, 166.02200212432226, 160.28814170216242, 128.22814671831827, 125.54741164443534, 135.31750958973134, 143.69526616741908, 154.56901598755564, 157.95032368285828, 151.89286401191887, 152.56006488288872, 159.3731242166705, 153.00260365378503, 165.6339463317645, 146.5479400491184, 154.6642780262972, 152.64958153220508, 128.74624691993225, 147.54062362530163, 140.72669055827936, 156.86166779138142, 143.810010649107, 166.38986741746334, 155.60304366513563, 157.6276797222233, 161.78385801885065, 160.5887550792144, 150.1163014740637, 161.05323973145764, 143.1463778376328, 151.27230262782663, 130.55503462123116, 166.25402215402397, 151.27230736462377, 147.1094671638239, 145.62279229095932, 163.22729860939774, 151.52603898555918, 140.9453257849222, 166.20806577965885, 167.34370447285983, 159.4600941247345, 147.13093590194242, 154.81374609277435, 165.73466905937457, 154.49581456007684, 142.88615078819754, 159.75701300316675, 165.97638193101466, 147.2968343694425, 160.6464423714282, 128.50240564215278, 147.87239918114113, 163.75297912852216, 143.48603412958374, 148.29765031974367, 162.6897449052678, 152.47117750536728, 155.59906998076227, 159.0077396634445, 159.70951346246534, 162.3136295534492, 164.05814681366107, 158.39173278445915, 151.66242256246647, 145.34744512225768, 164.4243161935072, 164.17372243329027, 155.90184940811085, 143.07034246086494, 161.56997139285957, 148.71066434034645, 156.1930436361133, 154.67104286815191, 170.58823529411765, 157.0443717135269, 164.91203763422206, 153.52900246811578, 142.85988927107445, 154.77614213295917, 147.7389427581243, 156.7812610691241, 164.9090165509373, 174.11764705882354, 143.87247907632417, 136.63837171100408, 155.77529631484722, 167.34829010504544, 132.88891600329333, 158.2753854761853, 146.02125961502773, 158.67846914433392, 158.37007085880433, 165.73529411764707, 157.8254527080125, 155.24136507005224, 155.96040484609807, 130.3619103388134, 154.90784363988283, 160.36833291730332, 155.97877953410753, 158.4209212517831, 129.21867283125985, 150.64021592965656, 158.5274761310889, 155.930665831207, 158.18179209208137, 161.88239932662378, 155.35454750626405, 152.57111822339633, 165.41263563356787, 148.11255829359078, 148.60792294405798, 150.82869962771437, 154.27825775334668, 134.5018761548888, 125.8142506624745, 162.4098638522239, 157.66993119773645, 159.8608353269245, 161.6553272493836, 157.56420013729866, 156.52047196566724, 156.42078125681965, 154.6606694329337, 143.33951486103456, 149.72984900196724, 155.18449945139906, 165.72610645528633, 141.97294026470476, 122.86228608793499, 166.84296112387415, 161.59970952059857, 157.6882858099867, 164.42114454916504, 165.1457046916506, 152.39215963242088, 145.82529897247605, 165.66559984257444, 165.08917442202488, 164.81016936133616, 155.87497062589588, 167.2058823529412, 154.02362717219097, 170.58823529411765, 156.75754604844207, 135.93876046435733, 155.81053627700004, 170.0, 154.72473841487968, 157.4215249619634, 145.8838599487082, 163.22797743737155, 157.33933456207555, 161.95282965695193, 162.69758767985473, 158.72559577432025, 165.0181464100203, 145.3692408405999, 172.04946275979668, 163.7627159711277, 170.2941176470588, 139.86958752553164, 147.0070877384447, 165.58945378237516, 167.26237728061403, 158.02150937792936, 161.07894210058683, 147.73957635673736, 172.2058823529412, 139.3477647002792, 163.55321985350915, 163.73708308844817, 148.91172860031068, 166.91176470588235, 154.6743884136784, 168.08310774675155, 162.15169266956937, 144.4999000711531, 139.44810634724024, 165.45755652759308, 162.6254050625724, 155.54600388844094, 138.33794682181107, 154.74916093985507, 165.02269345085625, 166.76470588235293, 148.34814694273908, 161.67680112046497, 170.3179257358621, 159.74977338643737, 147.64426484205143, 165.67082679894588, 166.8790510222752, 171.50089448540055, 163.66066865712492, 162.5923439400593, 166.74574977040453, 145.05753766109203, 162.61730742843682, 161.2361156774749, 162.3151353264049, 157.63652645498914, 155.1870561679398, 161.01944855197848, 160.03963114466902, 150.141534276512, 156.7954303080045, 159.11764705882354, 162.41807610545948, 159.96183460119101, 163.67465450369085, 166.93385058994286, 160.2957860208501, 154.97871470756365, 138.39769893141096, 164.99720121116619, 148.0126090989411, 164.51253326538085, 145.33290306414233, 142.9839973098925, 169.29393975093987, 142.16281300014586, 174.62537475587638, 156.0714389602935, 144.7435190909165, 169.55882352941177, 167.0192627085763, 146.87280276896558, 149.7346921441671, 158.90660960646474, 168.74181107711428, 139.7376561881282, 171.1764705882353, 157.06751332891636, 162.71695915917383, 153.79807944582942, 163.77186176555747, 168.31301538303032, 165.86318739569802, 162.28079454377914, 164.71676755888097, 166.72268315708288, 143.02831125288188, 165.58532637184896, 171.21035262133836, 157.52668679265938, 162.07443683112223, 165.0899770226873, 164.40662460124923, 161.59227439886266, 132.76384352970427, 163.50576351449686, 167.4264705882353, 169.11764705882354, 167.86915639332037, 168.38235294117646, 168.80174484776722, 151.49166922068358, 170.18390615804336, 167.58746716033258, 157.66457568491623, 157.25149535272317, 138.12143626580882, 168.07887452450257, 161.5300998364893, 166.0699917350318, 158.2350465875921, 169.58629577297185, 159.57627147570955, 162.83642118651423, 139.6333153154388, 149.8693558158115, 155.01950093595363, 169.5302826923914, 144.07809964849216, 158.8037480453802, 158.10607342067465, 170.22077121653246, 159.32394155782484, 132.7106307592793, 167.2535790145009, 168.48510870137665, 152.14820808080023, 168.4371092222311, 160.59337494386327, 161.8415571449734, 168.8235294117647, 160.30922755374044, 159.2294429568951, 131.02979769512513, 164.64010461129286, 161.0202029840504, 157.3802041811654, 171.12170086303595, 169.55882352941177, 166.66742753092004, 168.8235294117647, 144.7025023260629, 144.8131341533677, 167.1711426123198, 152.21166700087917, 166.052968071237, 169.97349170359928, 169.2122073640822, 159.71112638674495, 168.97022020126337, 137.82461366764178, 167.99564782111017, 157.41948139744335, 150.24700802333268, 156.4809278224406, 166.53525957429915, 168.39280408998025, 154.56690977299004, 156.12276533634812, 166.5195839583286, 161.10993265418824, 167.08203210820665, 170.31489056590732, 156.08832121506583, 170.3696242877329, 159.010466264979, 155.86947929602394, 160.02009229812117, 164.19501160728032, 159.8783357945119, 168.23529411764707, 152.98672267578, 166.48873464531442, 165.88235294117646, 160.66538386446888, 168.08825436778525, 164.23084147128088, 161.82486663994598, 162.9114005729334, 167.56222604204237, 157.61846519469916, 163.3793667567612, 155.64279676248893, 168.06060018008452, 162.77540524402235, 154.43314271475907, 156.1794209179386, 157.06990321828914, 153.6389569059772, 169.8607861852615, 150.2165235341726, 155.69971559137792, 159.65111015339548, 156.33385641973175, 165.1652052335497, 164.6021146173256, 168.0971710401471, 141.32392920083316, 157.69896905811734, 164.0565204132688, 157.1182494702078, 152.77472891142892, 161.21634925809127, 167.51045114880375, 152.55827154471694, 160.5309687190362, 172.62282893509212, 165.45273772356214, 157.57442501356516, 142.76416991598967, 156.83836245142925, 142.9462109766551, 153.0723167783332, 167.05882352941177, 158.4850036090659, 166.3235294117647, 165.8312379099125, 160.9777211733153, 160.21014866015696, 143.36423063106278, 172.64705882352942, 163.99806245160477, 144.56755093361443, 166.46518256888123, 126.6450969081838, 169.9025699101591, 157.43657813141533, 158.5057523476014, 165.46390779173691, 162.27168040715847, 157.0168885402813, 167.87873410786784, 161.1812062897737, 155.10513040391484, 165.99588439778745, 160.87491434152224, 151.15638238496268, 153.42338366780254, 138.50049643119002, 158.42667412951528, 159.86602512880359, 154.69993387770157, 157.34142524469317, 168.0835842226925, 170.58823529411765, 159.10484012868105, 150.73751045799318, 149.20517782323196, 160.217681332785, 147.30472665488628, 169.53854781227972, 162.26233989106018, 174.00750776299466, 169.82300465825412, 165.258767817416, 143.8861164164423, 158.1979236061116, 159.24536653918105, 155.99410765795315, 165.30853883582114, 163.42746352208334, 155.82361558889147, 164.69890270221043, 136.5548010768459, 146.1103457557749, 162.15263434131057, 158.0310899039287, 161.2929027614325, 161.05804426540493, 166.53706086837568, 152.47781380756277, 174.11764705882354, 155.81697540545028, 168.45298797719929, 137.3491023338615, 168.26906464005506, 167.6226907062944, 169.1819051460709, 154.04105138436842, 161.10947545082192, 166.8227851671514, 172.6334494492947, 166.82399590473707, 161.15137188824346, 155.3374076210001, 157.74307130229158, 166.87215451020705, 152.19528158086894, 160.69113522183136, 157.50914244961416, 161.8706124313041, 152.11264559335032, 152.41664895073725, 154.5126263591162, 160.9094102551739, 162.4475364802007, 162.36490241438491, 146.79063341582804, 165.0488506338144, 143.08426502710046, 164.20073474336846, 160.98856922754428, 149.0421833685624, 157.28929243773337, 149.82756033693732, 142.12712870530007, 161.69243067226628, 168.8045732998163, 173.97058823529412, 146.0141743497555, 148.41231377469052, 165.65357578299663, 169.8941176470588, 167.11051619898114, 165.6957182095287, 156.09925638783523, 161.64836901803162, 154.00212662302243, 151.23647424630653, 168.72228817711706, 148.17849044735146]
fitness = [int(c) for c in fitness]
fitness2 = [96.74373085160795, 84.87645619936141, 76.1056860213984, 109.18317049095856, 78.51178844360926, 87.4367515998319, 77.46973590212043, 83.26894009086534, 80.6423934440556, 93.01319673221154, 82.64922264335675, 83.44955026784602, 93.7482072708343, 103.99510465845704, 103.31937553530399, 85.72568813899214, 87.61731532338169, 101.9924433278704, 99.67432814621263, 74.09183691075131, 100.66814156708888, 102.14330365152212, 85.41657587026683, 94.98399072397095, 83.02667930358393, 118.7718265486176, 91.30373914920415, 93.45069525065388, 102.85106833351331, 96.53504238989626, 107.498052373983, 106.0432806554865, 86.7282469461182, 92.27387291603812, 80.58185220296976, 110.78885264476271, 119.33530893187186, 105.81503117228883, 113.80151156989984, 101.91535336347657, 104.24279720380113, 83.8977330028961, 109.78304747717532, 119.83929486934382, 103.15457260930293, 108.96229559826281, 106.93215624286634, 135.74788949479856, 112.23639595660721, 121.35107189309576, 108.89614832880606, 103.83563065330134, 126.07292196207612, 125.21706481593174, 123.68615807972826, 140.13459399477784, 122.44389424875168, 118.51932415827157, 121.89554844884223, 140.07987460972245, 135.20356049401101, 126.66758616942953, 120.71170572018148, 115.59369600855949, 124.16562142772375, 117.33749246665705, 119.15275366457301, 136.24379710888056, 145.84390239433634, 133.902154418547, 154.5444194149652, 135.21628824540744, 162.28335521353875, 141.93811871031403, 152.86910913864426, 151.88431953301745, 148.05417584839606, 137.0635425914774, 140.8940861175734, 130.37459805371697, 158.88279473095466, 156.71828166534956, 133.03578061709342, 137.36651680060487, 159.76299421640766, 137.4046042495054, 136.35204207556154, 121.70411949864923, 138.94284744242074, 142.77278059502385, 152.3409937441021, 153.3055181877957, 126.90321986201583, 144.28715850823477, 151.0834012688785, 130.62118959655731, 126.06814579217429, 159.65773871781215, 152.1040930601092, 158.13732991996577, 133.73865760663034, 138.69472042018268, 124.81506735542568, 146.6947446479666, 138.33094816144728, 141.0629208264777, 142.97167193821838, 133.37103863958862, 153.64421508788743, 147.41574752559077, 143.39058720177275, 149.90532830768336, 157.533189055852, 163.15896104011588, 151.7603081735996, 143.3402742605851, 131.7287937609533, 137.08956029901742, 162.96807272657426, 158.77190717156523, 142.3677899797986, 133.5152575704881, 150.80836438762339, 161.6961636677122, 144.53827019133814, 157.48087926898097, 142.52723963372023, 150.21054031750268, 145.01914716608778, 121.10024614809508, 158.33433524125599, 147.8488731286825, 142.64848711171265, 147.1270869503911, 143.8534950455917, 158.53642118651422, 139.74129974125424, 157.84385924850267, 152.18702190602676, 134.33474263042365, 131.9799377672784, 150.74238614823471, 114.72896349685567, 159.66759513014009, 163.07655673958544, 144.99702095809835, 136.5939406269188, 147.37657401649497, 146.98478111298104, 149.81531472782322, 142.70843216655896, 141.15137637887676, 153.0725238326144, 160.09200921260745, 140.25725677174046, 154.03190354152517, 146.54548233219057, 150.48178613635966, 155.17390842608532, 153.78710739063604, 154.2385270954012, 150.33886210014543, 149.72247059347794, 130.53472337406703, 149.93509299687238, 153.36149131723184, 159.70125069441133, 159.60093166838692, 143.24969362776878, 152.8684753751644, 141.3986740849547, 141.55962784077423, 148.61299633403155, 152.04392191506446, 150.11190454427717, 148.00627026474564, 149.0047636804787, 141.63667754451805, 148.13084900050083, 144.44462299047873, 158.7474298782894, 160.98953429142327, 144.55883574084604, 153.53475637298493, 137.3212884419921, 156.6257404585318, 156.2942995308962, 152.08587565769375, 139.1519594300668, 160.85085991175234, 159.74290283866165, 156.52238238401515, 140.01315714717208, 134.10442180298105, 153.55222829377325, 140.08524424361016, 160.19513936261217, 153.5580215193617, 140.19444644804966, 149.10936456130838, 129.5137017255413, 129.5465602559976, 144.9194131666916, 153.7037853782943, 157.94798845221462, 137.22123643275663, 164.11764705882354, 152.9893194653526, 171.08181082552014, 161.81512778227454, 159.4604977040637, 167.36171566294075, 149.49801629148672, 168.45209492178878, 173.52941176470588, 169.95937216089993, 161.93719025837242, 157.8762612563827, 151.3766573354049, 144.47669422431423, 159.12229688531644, 152.65507377693655, 150.72950836468004, 169.58263161821503, 174.30162541005348, 155.73034719806117, 152.5072094741565, 141.1203566368599, 166.35829509172467, 164.65926382204336, 171.92114454916504, 157.1876552088515, 158.20040737126848, 169.11251951145744, 160.58051971533473, 159.21482427673223, 157.22044032329563, 157.87469246051936, 150.75700810881705, 159.67646565078613, 140.27873239311816, 163.11930752034434, 166.66992116386743, 153.10828643582667, 151.60082748936478, 140.44796514935712, 168.8183893071316, 149.26069311647558, 148.8989326599581, 164.58400743453487, 169.25546917874217, 161.07382545643875, 162.11489516156416, 159.90906583511205, 145.0745028465759, 164.52596280124374, 147.9196286711201, 150.1028711317424, 149.28566916729596, 167.64705882352942, 160.56113388761528, 161.82399590473707, 171.76470588235293, 157.97262328157763, 165.4517524725581, 141.80954741288352, 169.48529411764707, 160.01340952701685, 153.76495783116715, 167.26795176878923, 168.97058823529412, 154.95884589897912, 173.38235294117646, 154.9409930332701, 158.0463236848364, 167.00706306041462, 160.72380933736463, 153.2168779768119, 160.32093756366476, 149.9618280981934, 155.01069808606903, 137.3893380686191, 164.04284801898672, 168.3175991541934, 162.88212680310457, 156.1672863732771, 173.97058823529412, 158.0165616004974, 160.6305082410955, 151.79307938176007, 164.5282754715971, 155.5229851110073, 163.90480263934217, 168.5149108904577, 154.50049005904194, 159.43111772502598, 172.35294117647058, 159.77660809893942, 148.642782683736, 165.73015401301396, 151.60141211728407, 143.88416974623777, 161.4464920187126, 166.98529411764707, 159.3129817258068, 173.20675328062669, 160.65210625724018, 166.05164618651435, 167.19480614586493, 163.34010246594636, 157.9984109422005, 174.41176470588235, 136.88077590812298, 156.48748665373077, 166.4898821176705, 149.45969892414342, 159.5959149049801, 166.61764705882354, 152.7112032993989, 155.0411628191711, 162.31632604954598, 165.53428137501217, 143.75482344553507, 160.77733212309826, 157.43458918664703, 169.2130382361457, 173.62344075454408, 166.55925280408474, 166.0984627563836, 156.07407785745946, 153.27403997398685, 163.5086009473564, 152.82464433087574, 156.1771448911269, 167.23907028502714, 170.88235294117646, 145.47333283501607, 156.84725577419866, 155.75041304371194, 166.91176470588235, 152.68579552768676, 154.34154890924134, 164.21065522529906, 168.8765139415739, 167.31131124744667, 143.89802473549074, 173.88879738039154, 164.65782729079382, 160.16996573755975, 160.58255966111162, 162.7245041506438, 158.8527058527706, 158.5590416124949, 162.27558411646004, 171.36044893946524, 170.88235294117646, 132.60417700658877, 170.63729810164, 163.23315627070158, 168.06430288691553, 171.14792975121492, 167.5637204746115, 159.05596396163358, 157.65631331727295, 163.01865784377668, 153.45828427054713, 173.97058823529412, 157.44599148575426, 161.4158248290081, 169.56759474356903, 154.94289513633288, 163.97010699866794, 156.1629400693858, 166.84837622032154, 157.3678734022667, 160.85959620666142, 160.4523937821612, 163.31470588235294, 161.18712528205512, 156.0233300311033, 154.4600334651043, 147.11997868929544, 160.35701631066397, 173.90855820091878, 168.38235294117646, 142.6838233098817, 158.23612375434968, 151.34142968075395, 143.7521910893932, 154.29030885647455, 154.1212674729321, 165.68404266868652, 154.2228034694881, 170.9519005477317, 159.7715474470798, 137.10019046827216, 152.60898863453622, 162.34349119024978, 164.2536232565955, 147.40260534694616, 162.45824876089713, 151.29869082495415, 164.70975373578415, 167.89954654156432, 153.3641012506352, 165.63124800221482, 158.36817565551652, 139.12314891303163, 165.97215719250272, 162.68584849467274, 158.49233822628278, 168.23529411764707, 167.19043002283678, 143.83177812135136, 143.26328724153828, 158.50330349321888, 173.6764705882353, 169.22407804325286, 158.3336185434565, 150.61532091752537, 165.15143340238643, 171.3235294117647, 147.9765940831601, 152.82167614133306, 167.37591679043211, 148.32629352109618, 172.71361004999403, 164.23160084341268, 158.40428697541472, 151.39888846831204, 159.0215844794927, 150.50418963030842, 154.62084679022996, 171.04754702079111, 159.51257666296425, 154.19972448346803, 165.027216427257, 157.33774418976213, 169.85294117647058, 148.39062831201514, 163.47359608932052, 152.83480957869935, 167.59454627568516, 165.90312586002497, 166.1764705882353, 159.8834797372136, 158.11294283076438, 154.63295066515204, 165.36985897709027, 159.0307878726596, 146.4037169724926, 161.96981964688146, 156.02368693616773, 155.5328600973327, 164.03619750704306, 155.89227943908864, 152.00180968117255, 161.09970319615152, 143.6294651103603, 167.27928004394028, 163.21081778559838, 141.14848915140513, 154.65755360702047, 162.1359873230135, 161.9949614997096, 160.07498978974468, 168.11814362401662, 155.53535934316758, 161.32951566034734, 152.4238179826356, 159.50843602776035, 165.53153578585614, 146.91540698156535, 166.55501025473436, 165.78700645328226, 140.4881610214772, 140.30447080311245, 169.9090165509373, 161.23601027367, 168.5202122546501, 158.20171564176076, 161.59959310214418, 169.2950121324594, 152.16497160355533, 158.13676800511374, 161.807560492164, 163.52339164070088, 155.8322951221576, 141.03602827350645, 165.40257937330352, 169.20888039995222, 155.980858038862, 160.08799013702966, 136.91261070789383, 160.40743025026748, 156.857158112794, 163.69302216300883, 165.28711980907246, 167.46728631639286, 170.25344052766735]
fitness2 = [int(c) for c in fitness2]
fitness3 = [96.72094299120016, 96.13693425788718, 85.99570644131656, 75.3987150502372, 99.7095914388733, 94.36518735752995, 86.48328163732963, 87.111227875721, 112.41295268781084, 125.16459122128784, 127.97458943578292, 113.74244908061173, 134.73904797507453, 153.62003876190397, 125.01760141825383, 142.07093497264495, 132.17467677740632, 127.6787404547611, 151.72214788312024, 157.43862960593137, 150.55748318728968, 166.02941176470588, 146.2983603626307, 159.10299906528883, 146.1499789161237, 132.82059967810656, 153.36340377765967, 156.36155558761754, 144.05239509765266, 144.3531747608182, 153.10825639886187, 155.20694552885652, 146.84502432339164, 137.72912028131086, 159.55368342477865, 157.8891327075866, 142.67020079071446, 142.63596340990318, 152.81981620737966, 147.04751001412632, 156.47182358858456, 149.28167867884386, 151.48326153845622, 148.87235898437584, 154.55670013065674, 157.55070284814664, 149.45813082481408, 170.0, 140.32594575049393, 148.149965483495, 130.3506963241258, 155.08881849103844, 159.22928090798854, 150.01691532150716, 158.267921127552, 141.40692396198295, 131.72446358256403, 159.0849537963829, 163.69967916703018, 139.66395800580702, 154.22875613977686, 141.39673005085018, 160.14882423661552, 155.1966766720491, 150.7423724792152, 162.60293870234642, 158.74187236583884, 138.29802522690335, 157.73646153138733, 142.75652201228354, 114.36277173083111, 167.37321268897665, 151.51851120269598, 163.71331792274924, 145.21604081123883, 172.5, 172.6652052335497, 158.79589172944742, 160.63922725999282, 148.34671239712335, 157.42468532121734, 161.99504931544573, 154.37049749938487, 162.2256120446429, 158.5528740405782, 175.6216582741008, 149.17099724254757, 135.8883383575347, 156.71335259299656, 174.11764705882354, 155.56414797093973, 168.95000000000002, 169.60995986278618, 149.18736913983133, 129.69552710628972, 153.12321187952347, 164.71248068654495, 163.41797529702694, 138.47294675680862, 152.61021920630267, 153.05114490272476, 167.95162761939199, 159.77571789799998, 142.4595483384054, 157.22045914001586, 143.45305396932037, 163.3903676798275, 159.52409149803862, 158.241511085206, 140.89599645566736, 168.18203610266218, 166.1858458797772, 138.72214473370644, 149.96298992308948, 160.75080280639108, 153.7071562238157, 155.62147273388612, 157.26761267574946, 160.7922132993381, 152.4376779455537, 162.85198443119086, 164.59076329638924, 169.05239061453463, 162.72592809335876, 163.86880466847262, 158.63998030908593, 159.45440056910763, 164.03950785173586, 155.70724044462165, 142.37064199652835, 172.5, 148.50626349757977, 162.37647715186327, 167.22436258696078, 159.94576666337414, 158.44456143106945, 166.19217413757124, 165.8287347433698, 163.7642607069244, 160.88844691145945, 165.59761513740034, 159.1191489171011, 173.97058823529412, 153.55044677217532, 144.35170917821986, 165.10822870110275, 150.90193317784062, 156.14926109333112, 158.50857126699728, 153.2581349157762, 158.19209782355998, 156.59128040402248, 144.16889365368587, 152.12831207212258, 164.13578231490877, 146.70251595080026, 173.9400715316266, 157.41476166356964, 163.07608372218832, 155.01458581277976, 159.89643057109603, 166.15153206675862, 158.08856304443253, 156.14567086392574, 165.764707865776, 151.90401729403987, 146.40073140882905, 148.01659138846722, 173.08823529411765, 148.5245559037691, 154.31441032093676, 165.69949176832728, 144.17253237557557, 164.61234555958578, 139.86667102515852, 168.19466627854698, 158.87267046301062, 165.62134837987765, 150.5695540071073, 163.49119872975297, 155.22264848249173, 165.73529411764707, 173.54754702079111, 158.9614449253087, 155.996765953546, 158.67527050512757, 155.5087826144356, 163.7425452628084, 151.26703674714906, 159.2562097286543, 150.19575203222786, 156.69940249314834, 144.07302747400098, 169.61808971702337, 162.40686898058314, 166.185850431518, 161.25337324334748, 144.8000653297015, 140.05602148261957, 156.12118612336891, 153.08387367561434, 166.20935394084557, 128.37625044819825, 152.0186171973001, 160.30761516153623, 162.45935394084557, 162.43890726684342, 162.99967947408211, 155.42167316873966, 159.563703764239, 160.77300438985287, 138.03646508950655, 167.1022294163085, 157.45733229036173, 145.9476222293149, 161.19333557335509, 151.31758707615643, 160.24868942499205, 153.9885239048113, 146.35474521176974, 148.54546797858018, 152.01811358476766, 162.43823529411765, 153.24981985524641, 153.25336573310202, 160.45260085442737, 160.63464687311415, 160.03802617585285, 147.4506369808713, 147.96925671897938, 165.06485031093027, 146.48101412236406, 162.32203058396837, 143.9279516392126, 170.44117647058823, 144.23012335037154, 144.64165801701463, 166.17817445745732, 168.39981351178855, 166.1301403885671, 153.63530209279932, 160.1853940452253, 147.78067710741027, 157.01830231467463, 129.13117845072006, 142.4948849049806, 160.13184189736594, 157.51629383001676, 154.05315923032038, 158.98361564732096, 153.74030952768598, 165.0858276747158, 154.2308433618616, 158.52941176470588, 152.93335806552182, 153.6181831593959, 139.94653441060146, 162.80969827652984, 151.73486308685318, 150.9870970178374, 152.40501145827986, 165.11320323963997, 140.52786352797494, 162.84314693536484, 164.82105240082933, 159.4400367603144, 164.7058823529412, 151.59954164490748, 147.30493719736066, 158.41648993488184, 145.11332689724048, 157.006112125108, 149.00804438775637, 164.96526025937862, 128.8785606123913, 159.99044656939256, 140.44380949074147, 153.30179977837466, 163.00923354725845, 160.90604848521218, 151.86581611613778, 160.0612513665969, 148.2891402558808, 162.0161207828341, 161.46234420165567, 162.88190102928914, 141.064600483753, 143.81153762587493, 156.69105215218937, 142.7957705412947, 156.05900192579344, 169.01239073486965, 141.7116199034813, 162.57220658010598, 163.2263600609473, 166.91888830145896, 155.52154052049218, 144.01737382702925, 162.44551046026942, 155.00291900161193, 153.45604243612235, 155.6182092079751, 150.64049602075156, 154.69046544584907, 162.98635559153797, 140.82997965071928, 147.34688912299708, 167.94117647058823, 166.4608063842276, 149.90028324693876, 156.03340540653352, 169.26470588235293, 139.8634036204742, 164.74272968745512, 143.97149253935402, 147.66091810156206, 163.8472132543192, 150.3031794818867, 159.93667322785603, 134.0342973425824, 152.09226447807697, 151.01356170912672, 148.06797549771952, 156.953405680804, 166.16394219630195, 166.32996614173157, 173.38235294117646, 145.8514420322949, 159.42701783302218, 140.19219596180332, 145.89551029630047, 152.00751881942767, 158.92831755364588, 161.5265735562891, 136.9390707413598, 167.64705882352942, 169.26470588235293, 159.95954334697979, 165.74079169694846, 169.57914972651122, 162.7941176470588, 173.75588235294117, 163.2953344886655, 167.42112500180474, 155.04234491381536, 151.94484015117192, 152.81813135960917, 151.8684382625122, 154.1218243242143, 159.14749730645403, 162.70351043889943, 163.02567967391911, 163.5725621303966, 156.30628412357487, 165.34285064524016, 159.86586913517, 145.09608183629263, 162.93181256287667, 168.52941176470588, 159.31108444894966, 147.5799429465476, 156.34173633098195, 158.67613582337685, 156.71994545882097, 146.55131914383406, 165.68226428395585, 151.30726135597155, 147.02323303671596, 169.86997710093084, 140.44883036015318, 169.0659794126163, 141.61203740115388, 154.02466499721925, 149.6784214822071, 170.82652745877576, 161.64808231182826, 161.39874849125314, 161.33268172240247, 152.91484875174982, 173.23529411764707, 153.88627630561902, 149.26276160638707, 169.8281447628872, 155.40023645916668, 162.5564386668121, 157.2231786501915, 173.36176470588236, 160.6357760804536, 162.98962127452404, 143.73961216792924, 135.8306535651976, 166.63898997923542, 168.58089974696372, 158.60280713657863, 142.99522488554965, 160.3407320370788, 157.25659084853737, 152.00150841300479, 173.38235294117646, 141.91927223531255, 154.2796575707534, 151.75578184703454, 168.64634706336733, 159.18203031299, 152.6267928259081, 152.09890576842133, 160.15498555755374, 161.44905927692662, 157.80811431429208, 156.38546288249213, 165.80503609825936, 166.1764705882353, 159.47824006459476, 144.05810047106579, 161.69472482172395, 153.77129096825576, 156.0646944009253, 166.71339339721536, 169.69480614586493, 171.47058823529412, 155.3603017614094, 168.14733175186913, 153.6295100484775, 156.5089262542782, 163.65735602105184, 156.23218942721863, 167.00299804701103, 164.7058823529412, 163.6366609064228, 154.25335004858457, 161.90360364052327, 161.70909628964083, 157.9805387822983, 145.70557410016684, 164.03078293663472, 155.36617890305777, 169.27734151592082, 156.0541684930638, 141.18232770681612, 167.78740903219455, 163.07644493602703, 164.48529411764707, 163.2188086507844, 137.9124951677667, 169.85294117647058, 166.29498857474434, 153.53550046393408, 160.7290765413693, 159.13841997767202, 157.4576663788069, 125.8739994479111, 138.5496388247077, 171.27186176555747, 169.41176470588235, 166.23074332129084, 167.64448443876276, 161.21978199614497, 160.62353972751745, 169.1177852876212, 168.21633800569865, 172.65406824533775, 154.8085807958856, 142.3308196996823, 162.5734627430831, 168.92217927703794, 158.7325146897173, 163.1421525083133, 162.1489173699528, 154.2906430566233, 149.97960335358914, 166.44293896279035, 159.83810351535467, 142.9228163532552, 149.03740635127815, 171.76470588235293, 161.56757797677105, 152.98455010765468, 166.95083781726592, 163.36601242598545, 162.3097688811403, 164.6966828428854, 159.8848172281802, 161.71793325143094, 165.28037275799184, 161.80202155426772, 163.11604709207762, 161.2584009672326, 162.62314480011423, 147.36472170839065, 148.17362419004533, 171.3235294117647, 163.87778433898416, 125.02676268672252, 169.49705882352941, 161.2429577432437, 170.9728176516026, 153.17124381049538, 150.35483873643997, 168.4378322986802, 157.51523214045824]
fitness3 = [int(c) for c in fitness3]


#plt.plot(fitness[::5])
plt.plot(fitness2[4::5], 'r')
plt.plot(fitness3[::5])
plt.ylabel('Average fitness')
plt.xlabel("Round # (every 5th)")
plt.show()