import numpy as np

import treams
from treams import SMatrix


def test_init():
    b = treams.PlaneWaveBasisPartial.default([0, 0])
    sm = SMatrix(np.zeros((2, 2, 2, 2)), basis=b, k0=1)
    assert (sm["up", "down"] == np.zeros((2, 2))).all()


def test_interface():
    b = treams.PlaneWaveBasisPartial.default([1, 2])
    sm = SMatrix.interface(2, b, [(2, 2, 1), (9, 1, 2)])
    m = treams.coeffs.fresnel(
        [[2, 6], [2, 10]], [[1j, np.sqrt(31)], [1j, np.sqrt(95)]], [1, 1 / 3]
    )
    assert (
        (sm[0, 0] == m[0, 0, ::-1, ::-1]).all()
        and (sm[0, 1] == m[0, 1, ::-1, ::-1]).all()
        and (sm[1, 0] == m[1, 0, ::-1, ::-1]).all()
        and (sm[1, 1] == m[1, 1, ::-1, ::-1]).all()
        and sm.k0 == 2
        and sm.basis == b
        and sm.material == ((9, 1, 2), (2, 2, 1))
    )


def test_slab():
    b = treams.PlaneWaveBasisPartial.default([1, 2])
    sm = SMatrix.slab(6, b, 3, [1, 2, 3])
    stest = SMatrix.interface(6, b, [1, 2])
    stest = stest.add(SMatrix.propagation([0, 0, 3], 6, b, 2))
    stest = stest.add(SMatrix.interface(6, b, [2, 3]))
    assert sm == stest


class TestArray:
    # a = np.eye(2)
    # b = treams.lattice.reciprocal(a)
    # kpars = [0.5, -1] + treams.lattice.diffr_orders_circle(b, 7) @ b
    basis = treams.PlaneWaveBasisPartial.diffr_orders([0.5, -1], np.eye(2), 7)
    expect = np.zeros((2, 2, 10, 10), complex)
    expect[0, 0, :, :] = [
        [
            0.993848113127513 + 0.0899002995446612j,
            0.000925536246960379 + 0.000428793418535671j,
            0.0648101516796695 + 0.0331042013001149j,
            0.0109997848957572 + 0.0879501663140503j,
            -0.106944388611334 + 0.0220159623986055j,
            -0.0524654995298520 - 0.0486312544338279j,
            -0.0395282122841410 + 0.130130316412195j,
            -0.0880140178152169 + 0.0622348131127568j,
            -0.00776234664607976 - 0.0517964465530611j,
            0.0575011455260130 - 0.0136799639308035j,
        ],
        [
            0.000368195704317514 + 0.000342748682178508j,
            0.993848113127513 + 0.0899002995446612j,
            0.0581175078289385 + 0.0374556467864508j,
            0.0345238953291475 - 0.0352425203792430j,
            -0.0967629926372309 + 0.0135040926582279j,
            -0.0908064879383120 + 0.102875938960868j,
            0.000499439658478980 - 0.0587987541584765j,
            -0.0762544852103155 - 0.0277378127693338j,
            -0.0335123168033311 + 0.0956716786902019j,
            0.0426603055441378 + 0.0935964180542818j,
        ],
        [
            -0.0224128384298332 - 0.0219558215375650j,
            0.0559328006698058 - 0.00699542481577819j,
            1.05772968849184 - 0.000726557597365671j,
            2.53274425719925e-05 - 0.00472287135244330j,
            0.0950097693048440 + 0.00349868895171054j,
            0.118914958142697 + 0.00899909281310485j,
            0.0131669242176258 - 0.00563755958120132j,
            0.0708970939412025 - 0.000381470646103337j,
            0.148214837500923 + 0.00710113570445585j,
            0.0889369170444238 + 0.00384425322789858j,
        ],
        [
            0.0238202986243872 - 0.0369604188036947j,
            0.0210529527146176 - 0.0412166735686972j,
            -0.000788697136521723 - 0.00370639325882959j,
            1.05772968849184 - 0.000726557597365654j,
            0.123848936004041 + 0.00901746586507995j,
            0.114317640119331 + 0.00673516839987441j,
            0.0955034726660238 + 0.00316322908150022j,
            0.167145666030970 + 0.00750611259077218j,
            0.0804645553554549 + 0.00233768107977424j,
            0.0160363331632666 - 0.00274273413656558j,
        ],
        [
            0.0430316846420204 + 0.0379831882156353j,
            -0.0203418294470036 + 0.0219456449502097j,
            0.0751896193624516 + 0.00442989155304419j,
            0.0782133923506291 + 0.00591893221832093j,
            1.03459593766718 - 0.00515702950143613j,
            -0.00183503160716397 - 0.00808019295498379j,
            0.124150439267534 + 0.00578304089460873j,
            0.0515989281281407 - 0.00140448865784358j,
            0.00784420213394574 - 0.00521672677020705j,
            0.0618591931224174 + 0.000614358342229893j,
        ],
        [
            0.00564858860599587 + 0.0404747176671437j,
            0.00920899445095448 + 0.0447334649038496j,
            0.0814585950765597 + 0.00593101664188928j,
            0.0624903416680794 + 0.00230117670616828j,
            -0.00288578201244165 - 0.00811521904373132j,
            1.03459593766718 - 0.00515702950143615j,
            0.0560126542341310 - 0.00112183963492036j,
            0.00654258248991011 - 0.00663466702217734j,
            0.0470016864821933 - 0.000991905292032428j,
            0.118835847135738 + 0.00527095321442688j,
        ],
        [
            -0.0125246319270290 + 0.0344316752004385j,
            0.0281012830307396 + 0.0397415321359903j,
            0.118674655221182 + 0.00532939528085764j,
            0.0503374594115830 - 0.000270846971257591j,
            0.00706264585686750 - 0.00716205009690607j,
            0.0557004755423458 - 0.00151613006265248j,
            1.03862277988706 - 0.00446705555367979j,
            -0.00219508632640696 - 0.00741309160956165j,
            0.0632478885667531 + 0.00199649718609005j,
            0.0884978111762212 + 0.00671330882080547j,
        ],
        [
            -0.0265497773644562 - 0.000225515181900920j,
            0.0587585736901527 + 0.0178484263957422j,
            0.0678081697252380 + 0.00224591596986489j,
            0.00934861327221400 - 0.00400270886750944j,
            0.0604650443412712 - 0.00121101355036169j,
            0.134019034054728 + 0.00624272905651435j,
            -0.00109259930845912 - 0.00707642969716172j,
            1.03862277988706 - 0.00446705555367978j,
            0.0851592113836609 + 0.00670555592027250j,
            0.0892197306191960 + 0.00666247656250550j,
        ],
        [
            0.0516543462992672 - 0.0235435312762873j,
            -0.00754975039571915 - 0.0317339503514190j,
            0.0139162688635973 - 0.00238013424123935j,
            0.0771791179996622 + 0.00333602831486019j,
            0.156790798261392 + 0.00695444162689790j,
            0.0816163851501056 + 0.000810578097589877j,
            0.109047519088116 + 0.00814311515044666j,
            0.108165163540845 + 0.00820524413939098j,
            1.04961445458457 - 0.00230362147631233j,
            -0.000860293278980379 - 0.00459476766940031j,
        ],
        [
            0.0527996490125280 + 0.0184949045426726j,
            -0.0285856194386801 + 0.00428391331727052j,
            0.0698268347829847 + 0.00202863074072325j,
            0.128620271681676 + 0.00616233852801594j,
            0.0620135432262052 - 0.00130870967208923j,
            0.0103495598672365 - 0.00688289581238961j,
            0.104084608465414 + 0.00819576826939015j,
            0.0773038125972791 + 0.00244018966991458j,
            -0.000170308595664239 - 0.00584860919902162j,
            1.04961445458457 - 0.00230362147631235j,
        ],
    ]
    expect[0, 1, :, :] = [
        [
            -0.00121078149386474 + 0.0105360703601872j,
            0.00574422068756450 - 0.0635401672310800j,
            -0.0253030666878696 - 0.0386545650920068j,
            -0.0596653988558001 + 0.0285699966014414j,
            0.0771472407627700 + 0.113260839302541j,
            0.0965915305871392 + 0.0280326390510608j,
            0.0828503474463993 - 0.0159811851477396j,
            0.0137058856410130 - 0.0562231823657133j,
            -0.0528842309708006 + 0.0863036360333729j,
            0.0204951609529228 + 0.0986436718619743j,
        ],
        [
            0.00574422068756448 - 0.0635401672310800j,
            -0.000661251301840379 + 0.0106219091307033j,
            -0.0220649981336179 + 0.0847439476373432j,
            -0.0655632564494079 + 0.0241213868563773j,
            0.0637189572477666 - 0.0386451330452704j,
            0.105244474539718 + 0.0366893023741864j,
            0.0796899264978385 + 0.0740574570592570j,
            0.0225564259673991 + 0.133081869952488j,
            -0.0499871371768027 - 0.0210649741895254j,
            0.0190341862946426 - 0.0490341097345966j,
        ],
        [
            -0.0245827628858486 + 0.0160917420023266j,
            0.0538937733698720 + 0.0140324594495978j,
            0.231438229589043 + 0.00638916820920929j,
            0.145090620819696 + 0.00260965627664133j,
            0.00507997677212323 - 0.00719456856856990j,
            0.0601752768619921 - 0.00184342131048677j,
            0.140879648047038 + 0.00187268530683326j,
            0.217488706951855 + 0.00817409029398223j,
            0.0688108517057143 - 0.00161422199257952j,
            0.000990173849110084 - 0.00574797601967016j,
        ],
        [
            0.0181693792293622 + 0.0379448158081856j,
            0.0153402407233594 + 0.0416956181885270j,
            0.145090620819696 + 0.00260965627664134j,
            0.231960412306503 + 0.00614763611638012j,
            0.0324681584860382 - 0.00410554038048649j,
            -0.00257352651987766 - 0.00709017151818118j,
            0.0104660368480857 - 0.00570865503676905j,
            0.101081926273698 - 0.000356171120013366j,
            0.166327790978530 + 0.00561205420255027j,
            0.0818267504441693 - 0.000450667795838653j,
        ],
        [
            0.0473755551432816 - 0.0322697004667758j,
            -0.0161647630626784 - 0.0266528218522238j,
            0.00334122992276013 - 0.00473205072798172j,
            0.0213551336033234 - 0.00270031832500847j,
            0.392210524336285 + 0.0100974878590117j,
            0.305491103403305 + 0.00640630969791261j,
            0.171259614900905 + 0.00260580831465352j,
            0.0484512190966304 - 0.00405008024594859j,
            0.105334792170552 + 0.00140697466471644j,
            0.186818806541043 + 0.00662420664576758j,
        ],
        [
            0.0117256930581958 - 0.0404029972926602j,
            0.0153466642000930 - 0.0440224126696036j,
            0.0395788100380689 - 0.00121246507989009j,
            -0.00169267384497088 - 0.00466338613279675j,
            0.305491103403305 + 0.00640630969791258j,
            0.393597649339270 + 0.00920346768214949j,
            0.242005684945980 + 0.00762234046944796j,
            0.162757694766738 + 0.00277205637477603j,
            0.0226054595075323 - 0.00377236165510020j,
            0.131272365173277 + 0.00191860356292815j,
        ],
        [
            -0.00721608669716130 - 0.0374099470431149j,
            0.0334396370338332 - 0.0359829019677023j,
            0.100025588797294 + 0.00132962037487127j,
            0.00743096332661488 - 0.00405318716513839j,
            0.184872871147433 + 0.00281294142269681j,
            0.261242475850786 + 0.00822823272296570j,
            0.347693111910818 + 0.00838917180126063j,
            0.261515262046721 + 0.00606363984315974j,
            0.000335269023654312 - 0.00520749638947393j,
            0.0535169888222304 - 0.00104565119232894j,
        ],
        [
            -0.0253868129672900 - 0.00618870616494402j,
            0.0600913075294249 - 0.0101850472198475j,
            0.154418585446480 + 0.00580366437503330j,
            0.0717689129157641 - 0.000252884121204149j,
            0.0523025582544394 - 0.00437201709159085j,
            0.175695141848094 + 0.00299240437556698j,
            0.261515262046721 + 0.00606363984315973j,
            0.345851664065581 + 0.00982597279438987j,
            0.0147765521941797 - 0.00429933303722608j,
            0.00559783899725002 - 0.00529131427737523j,
        ],
        [
            0.0476295780888575 + 0.0291859500301806j,
            -0.0116254178759258 + 0.0275870909156167j,
            0.0597137951251432 - 0.00140081569929742j,
            0.144338623747696 + 0.00487011927007519j,
            0.138977644769547 + 0.00185634794661291j,
            0.0298254114955998 - 0.00497721528890375j,
            0.000409777691580894 - 0.00636478677968881j,
            0.0180604261665935 - 0.00525479732105590j,
            0.265840432074685 + 0.00752251297923123j,
            0.181591899527160 + 0.00422135498979611j,
        ],
        [
            0.0544398438798653 - 0.0113109471850463j,
            -0.0270611305150395 - 0.0105046589477177j,
            0.000859269096347043 - 0.00498807170547375j,
            0.0710089424946557 - 0.000391087797391372j,
            0.246486817856847 + 0.00873991032901030j,
            0.173199412645827 + 0.00253138586907244j,
            0.0654103618070671 - 0.00127803197301369j,
            0.00684187735905910 - 0.00646723197502180j,
            0.181591899527160 + 0.00422135498979610j,
            0.265817226111122 + 0.00715697837137751j,
        ],
    ]
    expect[1, 0, :, :] = [
        [
            -0.000661251301840380 + 0.0106219091307033j,
            0.00574422068756449 - 0.0635401672310800j,
            -0.0655632564494079 + 0.0241213868563774j,
            -0.0220649981336179 + 0.0847439476373432j,
            0.105244474539718 + 0.0366893023741863j,
            0.0637189572477666 - 0.0386451330452704j,
            0.0225564259673991 + 0.133081869952488j,
            0.0796899264978385 + 0.0740574570592570j,
            0.0190341862946426 - 0.0490341097345966j,
            -0.0499871371768027 - 0.0210649741895254j,
        ],
        [
            0.00574422068756448 - 0.0635401672310800j,
            -0.00121078149386474 + 0.0105360703601872j,
            -0.0596653988558001 + 0.0285699966014415j,
            -0.0253030666878697 - 0.0386545650920068j,
            0.0965915305871392 + 0.0280326390510608j,
            0.0771472407627699 + 0.113260839302541j,
            0.0137058856410130 - 0.0562231823657133j,
            0.0828503474463992 - 0.0159811851477396j,
            0.0204951609529228 + 0.0986436718619743j,
            -0.0528842309708005 + 0.0863036360333730j,
        ],
        [
            0.0153402407233594 + 0.0416956181885270j,
            0.0181693792293622 + 0.0379448158081856j,
            0.231960412306503 + 0.00614763611638005j,
            0.145090620819696 + 0.00260965627664124j,
            -0.00257352651987778 - 0.00709017151818119j,
            0.0324681584860382 - 0.00410554038048648j,
            0.101081926273698 - 0.000356171120013381j,
            0.0104660368480857 - 0.00570865503676909j,
            0.0818267504441693 - 0.000450667795838703j,
            0.166327790978530 + 0.00561205420255021j,
        ],
        [
            0.0538937733698720 + 0.0140324594495977j,
            -0.0245827628858486 + 0.0160917420023266j,
            0.145090620819696 + 0.00260965627664124j,
            0.231438229589043 + 0.00638916820920918j,
            0.0601752768619922 - 0.00184342131048682j,
            0.00507997677212339 - 0.00719456856856997j,
            0.217488706951855 + 0.00817409029398213j,
            0.140879648047038 + 0.00187268530683316j,
            0.000990173849110143 - 0.00574797601967020j,
            0.0688108517057144 - 0.00161422199257951j,
        ],
        [
            0.0153466642000930 - 0.0440224126696036j,
            0.0117256930581958 - 0.0404029972926602j,
            -0.00169267384497089 - 0.00466338613279674j,
            0.0395788100380688 - 0.00121246507989005j,
            0.393597649339270 + 0.00920346768214930j,
            0.305491103403305 + 0.00640630969791241j,
            0.162757694766738 + 0.00277205637477594j,
            0.242005684945980 + 0.00762234046944789j,
            0.131272365173277 + 0.00191860356292812j,
            0.0226054595075323 - 0.00377236165510023j,
        ],
        [
            -0.0161647630626783 - 0.0266528218522238j,
            0.0473755551432816 - 0.0322697004667758j,
            0.0213551336033234 - 0.00270031832500846j,
            0.00334122992276020 - 0.00473205072798169j,
            0.305491103403305 + 0.00640630969791246j,
            0.392210524336285 + 0.0100974878590115j,
            0.0484512190966304 - 0.00405008024594861j,
            0.171259614900905 + 0.00260580831465349j,
            0.186818806541043 + 0.00662420664576754j,
            0.105334792170552 + 0.00140697466471633j,
        ],
        [
            0.0600913075294249 - 0.0101850472198475j,
            -0.0253868129672900 - 0.00618870616494402j,
            0.0717689129157641 - 0.000252884121204201j,
            0.154418585446480 + 0.00580366437503326j,
            0.175695141848094 + 0.00299240437556694j,
            0.0523025582544393 - 0.00437201709159089j,
            0.345851664065581 + 0.00982597279438973j,
            0.261515262046721 + 0.00606363984315957j,
            0.00559783899724996 - 0.00529131427737522j,
            0.0147765521941796 - 0.00429933303722609j,
        ],
        [
            0.0334396370338332 - 0.0359829019677023j,
            -0.00721608669716131 - 0.0374099470431149j,
            0.00743096332661488 - 0.00405318716513840j,
            0.100025588797294 + 0.00132962037487128j,
            0.261242475850786 + 0.00822823272296553j,
            0.184872871147433 + 0.00281294142269667j,
            0.261515262046721 + 0.00606363984315961j,
            0.347693111910818 + 0.00838917180126050j,
            0.0535169888222303 - 0.00104565119232894j,
            0.000335269023654362 - 0.00520749638947396j,
        ],
        [
            -0.0270611305150396 - 0.0105046589477177j,
            0.0544398438798652 - 0.0113109471850463j,
            0.0710089424946556 - 0.000391087797391355j,
            0.000859269096346950 - 0.00498807170547373j,
            0.173199412645827 + 0.00253138586907232j,
            0.246486817856847 + 0.00873991032901021j,
            0.00684187735905901 - 0.00646723197502182j,
            0.0654103618070671 - 0.00127803197301367j,
            0.265817226111122 + 0.00715697837137738j,
            0.181591899527160 + 0.00422135498979600j,
        ],
        [
            -0.0116254178759258 + 0.0275870909156167j,
            0.0476295780888575 + 0.0291859500301807j,
            0.144338623747696 + 0.00487011927007516j,
            0.0597137951251432 - 0.00140081569929746j,
            0.0298254114955999 - 0.00497721528890372j,
            0.138977644769547 + 0.00185634794661296j,
            0.0180604261665935 - 0.00525479732105592j,
            0.000409777691580898 - 0.00636478677968882j,
            0.181591899527160 + 0.00422135498979598j,
            0.265840432074685 + 0.00752251297923111j,
        ],
    ]
    expect[1, 1, :, :] = [
        [
            0.993848113127513 + 0.0899002995446612j,
            0.000368195704317515 + 0.000342748682178508j,
            0.0345238953291475 - 0.0352425203792430j,
            0.0581175078289385 + 0.0374556467864508j,
            -0.0908064879383120 + 0.102875938960868j,
            -0.0967629926372310 + 0.0135040926582279j,
            -0.0762544852103155 - 0.0277378127693338j,
            0.000499439658479015 - 0.0587987541584765j,
            0.0426603055441378 + 0.0935964180542817j,
            -0.0335123168033311 + 0.0956716786902020j,
        ],
        [
            0.000925536246960379 + 0.000428793418535676j,
            0.993848113127513 + 0.0899002995446612j,
            0.0109997848957571 + 0.0879501663140503j,
            0.0648101516796695 + 0.0331042013001149j,
            -0.0524654995298519 - 0.0486312544338279j,
            -0.106944388611334 + 0.0220159623986055j,
            -0.0880140178152169 + 0.0622348131127568j,
            -0.0395282122841410 + 0.130130316412195j,
            0.0575011455260130 - 0.0136799639308034j,
            -0.00776234664607974 - 0.0517964465530611j,
        ],
        [
            0.0210529527146176 - 0.0412166735686972j,
            0.0238202986243873 - 0.0369604188036947j,
            1.05772968849184 - 0.000726557597365666j,
            -0.000788697136521705 - 0.00370639325882957j,
            0.114317640119331 + 0.00673516839987453j,
            0.123848936004041 + 0.00901746586508000j,
            0.167145666030970 + 0.00750611259077218j,
            0.0955034726660237 + 0.00316322908150025j,
            0.0160363331632666 - 0.00274273413656559j,
            0.0804645553554549 + 0.00233768107977422j,
        ],
        [
            0.0559328006698059 - 0.00699542481577820j,
            -0.0224128384298332 - 0.0219558215375650j,
            2.53274425721147e-05 - 0.00472287135244334j,
            1.05772968849184 - 0.000726557597365662j,
            0.118914958142697 + 0.00899909281310487j,
            0.0950097693048439 + 0.00349868895171065j,
            0.0708970939412025 - 0.000381470646103337j,
            0.0131669242176258 - 0.00563755958120129j,
            0.0889369170444238 + 0.00384425322789861j,
            0.148214837500923 + 0.00710113570445587j,
        ],
        [
            0.00920899445095450 + 0.0447334649038496j,
            0.00564858860599586 + 0.0404747176671437j,
            0.0624903416680793 + 0.00230117670616833j,
            0.0814585950765597 + 0.00593101664188929j,
            1.03459593766718 - 0.00515702950143621j,
            -0.00288578201244174 - 0.00811521904373130j,
            0.00654258248991009 - 0.00663466702217730j,
            0.0560126542341310 - 0.00112183963492035j,
            0.118835847135738 + 0.00527095321442691j,
            0.0470016864821934 - 0.000991905292032356j,
        ],
        [
            -0.0203418294470036 + 0.0219456449502097j,
            0.0430316846420204 + 0.0379831882156353j,
            0.0782133923506292 + 0.00591893221832097j,
            0.0751896193624516 + 0.00442989155304420j,
            -0.00183503160716394 - 0.00808019295498380j,
            1.03459593766718 - 0.00515702950143617j,
            0.0515989281281407 - 0.00140448865784353j,
            0.124150439267534 + 0.00578304089460875j,
            0.0618591931224175 + 0.000614358342229880j,
            0.00784420213394577 - 0.00521672677020706j,
        ],
        [
            0.0587585736901527 + 0.0178484263957421j,
            -0.0265497773644562 - 0.000225515181900931j,
            0.00934861327221396 - 0.00400270886750944j,
            0.0678081697252379 + 0.00224591596986489j,
            0.134019034054728 + 0.00624272905651429j,
            0.0604650443412711 - 0.00121101355036160j,
            1.03862277988706 - 0.00446705555367977j,
            -0.00109259930845908 - 0.00707642969716172j,
            0.0892197306191960 + 0.00666247656250551j,
            0.0851592113836610 + 0.00670555592027249j,
        ],
        [
            0.0281012830307397 + 0.0397415321359903j,
            -0.0125246319270290 + 0.0344316752004384j,
            0.0503374594115831 - 0.000270846971257560j,
            0.118674655221182 + 0.00532939528085768j,
            0.0557004755423458 - 0.00151613006265246j,
            0.00706264585686755 - 0.00716205009690603j,
            -0.00219508632640699 - 0.00741309160956164j,
            1.03862277988706 - 0.00446705555367977j,
            0.0884978111762212 + 0.00671330882080548j,
            0.0632478885667531 + 0.00199649718609007j,
        ],
        [
            -0.0285856194386801 + 0.00428391331727053j,
            0.0527996490125280 + 0.0184949045426726j,
            0.128620271681676 + 0.00616233852801598j,
            0.0698268347829847 + 0.00202863074072327j,
            0.0103495598672364 - 0.00688289581238952j,
            0.0620135432262052 - 0.00130870967208924j,
            0.0773038125972792 + 0.00244018966991466j,
            0.104084608465414 + 0.00819576826939018j,
            1.04961445458457 - 0.00230362147631232j,
            -0.000170308595664255 - 0.00584860919902158j,
        ],
        [
            -0.00754975039571915 - 0.0317339503514190j,
            0.0516543462992672 - 0.0235435312762873j,
            0.0771791179996623 + 0.00333602831486019j,
            0.0139162688635972 - 0.00238013424123936j,
            0.0816163851501056 + 0.000810578097589912j,
            0.156790798261392 + 0.00695444162689790j,
            0.108165163540845 + 0.00820524413939102j,
            0.109047519088116 + 0.00814311515044669j,
            -0.000860293278980314 - 0.00459476766940031j,
            1.04961445458457 - 0.00230362147631232j,
        ],
    ]

    def test(self):
        tm = treams.TMatrix.sphere(4, 3, [0.2], [4, 1])
        sm = SMatrix.from_array(tm, self.basis)
        assert all(
            np.all(np.abs(sm[i, j] - self.expect[i, j]) < 1e-8)
            for i in range(2)
            for j in range(2)
        )

    def test_cyl(self):
        tm = treams.TMatrix.sphere(4, 3, [0.2], [4, 1])
        # A larger range for kz (which later is kx) is needed for convergence
        cwb = treams.CylindricalWaveBasis.diffr_orders(0.5, 4, 1, 10)
        tmc = treams.TMatrixC.from_array(tm, cwb)
        basis = treams.PlaneWaveBasisPartial._from_iterable(self.basis, "zx")
        qm = SMatrix.from_array(tmc, basis)
        assert np.all(np.abs(qm.q - self.expect) < 1e-8)


# class TestDouble:
#     def test(self):
#         qm = QMatrix.slab(6, [1, 2], 3, [1, 2, 1])
#         qm.double()
#         qtest = QMatrix.slab(6, [1, 2], 6, [1, 2, 1])
#         assert np.all(np.abs(qm.q - qtest.q) < 1e-8)

# class TestChangeBasis:
#     def test(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]))
#         qm.changebasis()
#         assert (
#             not qm.helicity
#             and np.all(np.abs(qm.q[0, 0] - [[0, 2], [1, 5]]) < 1e-14)
#             and np.all(np.abs(qm.q[0, 1] - [[0, 2], [1, 13]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 0] - [[0, 2], [1, 21]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 1] - [[0, 2], [1, 29]]) < 1e-14)
#         )

# class TestHelicityBasis:
#     def test(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]), helicity=False)
#         qm.helicitybasis()
#         assert (
#             qm.helicity
#             and np.all(np.abs(qm.q[0, 0] - [[0, 2], [1, 5]]) < 1e-14)
#             and np.all(np.abs(qm.q[0, 1] - [[0, 2], [1, 13]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 0] - [[0, 2], [1, 21]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 1] - [[0, 2], [1, 29]]) < 1e-14)
#         )
#     def test_no_change(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]), helicity=True)
#         qm.helicitybasis()
#         assert qm.helicity and np.all(qm.q == q)
#     def test_pick(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]), helicity=True)
#         qm.helicitybasis(([0], [0], [0]))
#         assert qm.helicity and np.all(qm.q == q[:, :, :1, :1])

# class TestParityBasis:
#     def test(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]))
#         qm.paritybasis()
#         assert (
#             not qm.helicity
#             and np.all(np.abs(qm.q[0, 0] - [[0, 2], [1, 5]]) < 1e-14)
#             and np.all(np.abs(qm.q[0, 1] - [[0, 2], [1, 13]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 0] - [[0, 2], [1, 21]]) < 1e-14)
#             and np.all(np.abs(qm.q[1, 1] - [[0, 2], [1, 29]]) < 1e-14)
#         )
#     def test_no_change(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]), helicity=False)
#         qm.paritybasis()
#         assert not qm.helicity and np.all(qm.q == q)
#     def test_pick(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]), helicity=False)
#         qm.paritybasis(([0], [0], [0]))
#         assert not qm.helicity and np.all(qm.q == q[:, :, :1, :1])

# class TestFieldOutside:
#     def test(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]))
#         a, b = qm.field_outside(([1, 2], [3, 4]))
#         assert np.all(a == [44, 64]) and np.all(b == [124, 144])


# class TestFieldInside:
#     def test(self):
#         q = np.zeros((2, 2, 2, 2))
#         q[0, 0] = [[1, 2], [3, 4]]
#         q[0, 1] = [[5, 6], [7, 8]]
#         q[1, 0] = [[9, 10], [11, 12]]
#         q[1, 1] = [[13, 14], [15, 16]]
#         qm = QMatrix(q, 1, modes=([0, 0], [0, 0], [0, 1]))
#         a, b = qm.field_inside(([1, 2], [3, 4]), qm)
#         assert (
#             np.all(np.abs(a - [-6.41911765, -3.50735294]) < 1e-8)
#             and np.all(np.abs(b - [ 2.15441176, -3.69852941]) < 1e-8)
#         )

# class TestField:
#     def test_helicity(self):
#         qm = QMatrix(np.eye(2), 5, modes=([-4, -4], [0, 0], [0, 1]))
#         r = [1, 2, 3]
#         expect = treams.special.vpw_A(-4, 0, 3, *r, [0, 1])
#         assert np.all(np.abs(qm.field(r) - expect) < 1e-16)
#     def test_parity(self):
#         qm = QMatrix(np.eye(2), 5, modes=([-4, -4], [0, 0], [0, 1]), helicity=False)
#         r = [1, 2, 3]
#         expect = np.stack((treams.special.vpw_M(-4, 0, -3, *r), treams.special.vpw_N(-4, 0, -3, *r)))
#         assert np.all(np.abs(qm.field(r, direction=-1) - expect) < 1e-16)

# class TestTR:
#     def test_helicity(self):
#         qm = QMatrix.interface(3, [0, 0], [1.5 + .3j, 1.2 + .1j], [1.1 + .1j, 1], [.1 + .01j, -.2 - .01j])
#         t, r = qm.tr([.3, .2])
#         assert np.abs(t + r - 1) < 1e-15
#     def test_parity(self):
#         qm = QMatrix.interface(3, [0, 0], [1.5 + .3j, 1.2 + .1j], [1.1 + .1j, 1])
#         qm.paritybasis()
#         t, r = qm.tr([.3, .2])
#         assert np.abs(t + r - 1) < 1e-15

# class TestChiralityDensity:
#     def test_z_zero_helicity(self):
#         qm_below = QMatrix.interface(4, [1, 0], [1, 4 + .1j])
#         qm_below.add(QMatrix.propagation([0, 0, 2], 4, [1, 0], 4 + .1j))
#         qm_above = QMatrix.interface(4, [1, 0], [4 + .1j, 1])
#         coeffs = qm_below.field_inside(([1, 0], None), qm_above)
#         assert isclose(qm_below.chirality_density(coeffs), 0.2972593823648686)
#     def test_z_zero_parity(self):
#         qm_below = QMatrix.interface(4, [1, 0], [1, 4 + .1j])
#         qm_below.add(QMatrix.propagation([0, 0, 2], 4, [1, 0], 4 + .1j))
#         qm_above = QMatrix.interface(4, [1, 0], [4 + .1j, 1])
#         qm_below.paritybasis()
#         qm_above.paritybasis()
#         coeffs = qm_below.field_inside(([np.sqrt(0.5), -np.sqrt(0.5)], None), qm_above)
#         assert isclose(qm_below.chirality_density(coeffs), -0.2972593823648686)
#     def test_helicity(self):
#         qm_below = QMatrix.interface(4, [1, 0], [1, 4 + .1j])
#         qm_below.add(QMatrix.propagation([0, 0, 2], 4, [1, 0], 4 + .1j))
#         qm_above = QMatrix.interface(4, [1, 0], [4 + .1j, 1])
#         coeffs = qm_below.field_inside(([1, 0], None), qm_above)
#         assert isclose(qm_below.chirality_density(coeffs, (-2, 0)), 0.38113929722178935,)
#     def test_parity(self):
#         qm_below = QMatrix.interface(4, [1, 0], [1, 4 + .1j])
#         qm_below.add(QMatrix.propagation([0, 0, 2], 4, [1, 0], 4 + .1j))
#         qm_above = QMatrix.interface(4, [1, 0], [4 + .1j, 1])
#         qm_below.paritybasis()
#         qm_above.paritybasis()
#         coeffs = qm_below.field_inside(([np.sqrt(0.5), -np.sqrt(0.5)], None), qm_above)
#         assert isclose(qm_below.chirality_density(coeffs, (-2, 0)), -0.38113929722178935, 1e-7)

# class TestCD:
#     def test_h(self):
#         qm = QMatrix.slab(4, [1, 0], 4, [1, 4 + .1j, 1], kappa=[0, .5, 0])
#         assert np.all(np.abs(np.array([-0.003530982180309652, -0.0032078453784040294]) - qm.cd([1, 0])) < 1e-8)
#     def test_p(self):
#         qm = QMatrix.slab(4, [1, 0], 4, [1, 4 + .1j, 1], kappa=[0, .5, 0])
#         qm.paritybasis()
#         assert np.all(np.abs(np.array([0.003530982180309652, 0.0032078453784040294]) - qm.cd([np.sqrt(.5), -np.sqrt(.5)])) < 1e-8)
