-> subjects split into 70% for training, 15% for validation, and 15% for testing. The validation process was performed on the validation subset at each communication round during federated learning to monitor model convergence.

-> Results: 128 for 5 subjects(3-1-1 - (T-V-T))

-> No Master-slave

--------------------------------------------------------------------------------------256 - 50 ------------------------------------------------------------------------------------
=== Running FEDAVG with Overlap = 0.0, Num Clients = 3 ===
[FederatedServer] Initialized server with models: ['binary', 'fall', 'non_fall']

=== FEDAVG Round 1/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.7224, W-Acc=0.4443, Prec=0.5913, Rec=0.7224, F1=0.6349, Fall: Acc=0.6572, W-Acc=0.1752, Prec=0.6673, Rec=0.6572, F1=0.6482, Non-Fall: Acc=0.6221, W-Acc=0.1588, Prec=0.5608, Rec=0.6221, F1=0.5568
Average Client Metrics (Two Models) - Binary: Acc=0.7224, W-Acc=0.4443, Prec=0.5913, Rec=0.7224, F1=0.6349, Multiclass: Acc=0.6396, W-Acc=0.1670, Prec=0.6140, Rec=0.6396, F1=0.6025
[Debug] binary model norm after loading: 38.4054
[Debug] fall model norm after loading: 50.7961
[Debug] non_fall model norm after loading: 41.8051

Global Validation Metrics (Three Models) - Binary: Acc=0.3335, W-Acc=0.1112, Prec=0.1112, Rec=0.3335, F1=0.1668, Fall: Acc=0.4764, W-Acc=0.1527, Prec=0.3193, Rec=0.4764, F1=0.3820, Non-Fall: Acc=0.3721, W-Acc=0.0897, Prec=0.3465, Rec=0.3721, F1=0.3351
Global Validation Metrics (Two Models) - Binary: Acc=0.3335, W-Acc=0.1112, Prec=0.1112, Rec=0.3335, F1=0.1668, Multiclass: Acc=0.4242, W-Acc=0.1212, Prec=0.3329, Rec=0.4242, F1=0.3585
[Debug] binary model norm after loading: 38.4054
[Debug] fall model norm after loading: 50.7961
[Debug] non_fall model norm after loading: 41.8051

Global Test Metrics (Three Models) - Binary: Acc=0.3332, W-Acc=0.1111, Prec=0.1111, Rec=0.3332, F1=0.1666, Fall: Acc=0.3954, W-Acc=0.1045, Prec=0.4310, Rec=0.3954, F1=0.3155, Non-Fall: Acc=0.4229, W-Acc=0.1071, Prec=0.2953, Rec=0.4229, F1=0.3154
Global Test Metrics (Two Models) - Binary: Acc=0.3332, W-Acc=0.1111, Prec=0.1111, Rec=0.3332, F1=0.1666, Multiclass: Acc=0.4091, W-Acc=0.1058, Prec=0.3631, Rec=0.4091, F1=0.3154 
Results saved to D:\FL_MobiAct_updated - Copy\federated_results/overlap_0.0_num_clients_3\fedavg\round_0_results.csv

=== FEDAVG Round 2/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.8153, W-Acc=0.4713, Prec=0.7557, Rec=0.8153, F1=0.7725, Fall: Acc=0.7339, W-Acc=0.1984, Prec=0.7430, Rec=0.7339, F1=0.7188, Non-Fall: Acc=0.7143, W-Acc=0.1738, Prec=0.6770, Rec=0.7143, F1=0.6627
Average Client Metrics (Two Models) - Binary: Acc=0.8153, W-Acc=0.4713, Prec=0.7557, Rec=0.8153, F1=0.7725, Multiclass: Acc=0.7241, W-Acc=0.1861, Prec=0.7100, Rec=0.7241, F1=0.6907
[Debug] binary model norm after loading: 42.3252
[Debug] fall model norm after loading: 68.1576
[Debug] non_fall model norm after loading: 53.1264

Global Validation Metrics (Three Models) - Binary: Acc=0.4726, W-Acc=0.3143, Prec=0.3949, Rec=0.4726, F1=0.4299, Fall: Acc=0.4751, W-Acc=0.1516, Prec=0.3135, Rec=0.4751, F1=0.3652, Non-Fall: Acc=0.3297, W-Acc=0.0760, Prec=0.3830, Rec=0.3297, F1=0.2904
Global Validation Metrics (Two Models) - Binary: Acc=0.4726, W-Acc=0.3143, Prec=0.3949, Rec=0.4726, F1=0.4299, Multiclass: Acc=0.4024, W-Acc=0.1138, Prec=0.3483, Rec=0.4024, F1=0.3278
[Debug] binary model norm after loading: 42.3252
[Debug] fall model norm after loading: 68.1576
[Debug] non_fall model norm after loading: 53.1264

Global Test Metrics (Three Models) - Binary: Acc=0.5683, W-Acc=0.3765, Prec=0.4437, Rec=0.5683, F1=0.4926, Fall: Acc=0.4775, W-Acc=0.1252, Prec=0.5129, Rec=0.4775, F1=0.4740, Non-Fall: Acc=0.4464, W-Acc=0.1124, Prec=0.2820, Rec=0.4464, F1=0.3165
Global Test Metrics (Two Models) - Binary: Acc=0.5683, W-Acc=0.3765, Prec=0.4437, Rec=0.5683, F1=0.4926, Multiclass: Acc=0.4620, W-Acc=0.1188, Prec=0.3974, Rec=0.4620, F1=0.3952 
Results saved to D:\FL_MobiAct_updated - Copy\federated_results/overlap_0.0_num_clients_3\fedavg\round_1_results.csv

=== FEDAVG Round 3/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.8510, W-Acc=0.4950, Prec=0.7811, Rec=0.8510, F1=0.8071, Fall: Acc=0.7987, W-Acc=0.2145, Prec=0.8076, Rec=0.7987, F1=0.7978, Non-Fall: Acc=0.7679, W-Acc=0.1785, Prec=0.7463, Rec=0.7679, F1=0.7354
Average Client Metrics (Two Models) - Binary: Acc=0.8510, W-Acc=0.4950, Prec=0.7811, Rec=0.8510, F1=0.8071, Multiclass: Acc=0.7833, W-Acc=0.1965, Prec=0.7769, Rec=0.7833, F1=0.7666

Global Validation Metrics (Three Models) - Binary: Acc=0.6391, W-Acc=0.4260, Prec=0.4380, Rec=0.6391, F1=0.5198, Fall: Acc=0.4919, W-Acc=0.1525, Prec=0.3513, Rec=0.4919, F1=0.4004, Non-Fall: Acc=0.4205, W-Acc=0.0913, Prec=0.4059, Rec=0.4205, F1=0.3886
Global Validation Metrics (Two Models) - Binary: Acc=0.6391, W-Acc=0.4260, Prec=0.4380, Rec=0.6391, F1=0.5198, Multiclass: Acc=0.4562, W-Acc=0.1219, Prec=0.3786, Rec=0.4562, F1=0.3945

Global Test Metrics (Three Models) - Binary: Acc=0.6544, W-Acc=0.4351, Prec=0.5052, Rec=0.6544, F1=0.5339, Fall: Acc=0.5108, W-Acc=0.1258, Prec=0.5178, Rec=0.5108, F1=0.4983, Non-Fall: Acc=0.5438, W-Acc=0.1313, Prec=0.4929, Rec=0.5438, F1=0.4984
Global Test Metrics (Two Models) - Binary: Acc=0.6544, W-Acc=0.4351, Prec=0.5052, Rec=0.6544, F1=0.5339, Multiclass: Acc=0.5273, W-Acc=0.1286, Prec=0.5053, Rec=0.5273, F1=0.4983 
Results saved to D:\FL_MobiAct_updated - Copy\federated_results/overlap_0.0_num_clients_3\fedavg\round_2_results.csv

=== FEDAVG Round 4/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.8757, W-Acc=0.5119, Prec=0.8016, Rec=0.8757, F1=0.8312, Fall: Acc=0.8159, W-Acc=0.2192, Prec=0.8337, Rec=0.8159, F1=0.8075, Non-Fall: Acc=0.7859, W-Acc=0.1801, Prec=0.7874, Rec=0.7859, F1=0.7563
Average Client Metrics (Two Models) - Binary: Acc=0.8757, W-Acc=0.5119, Prec=0.8016, Rec=0.8757, F1=0.8312, Multiclass: Acc=0.8009, W-Acc=0.1996, Prec=0.8105, Rec=0.8009, F1=0.7819

Global Validation Metrics (Three Models) - Binary: Acc=0.6500, W-Acc=0.4331, Prec=0.4473, Rec=0.6500, F1=0.5257, Fall: Acc=0.4761, W-Acc=0.1484, Prec=0.3508, Rec=0.4761, F1=0.3911, Non-Fall: Acc=0.4515, W-Acc=0.1007, Prec=0.4686, Rec=0.4515, F1=0.4387
Global Validation Metrics (Two Models) - Binary: Acc=0.6500, W-Acc=0.4331, Prec=0.4473, Rec=0.6500, F1=0.5257, Multiclass: Acc=0.4638, W-Acc=0.1246, Prec=0.4097, Rec=0.4638, F1=0.4149

Global Test Metrics (Three Models) - Binary: Acc=0.6668, W-Acc=0.4446, Prec=0.7778, Rec=0.6668, F1=0.5336, Fall: Acc=0.5056, W-Acc=0.1242, Prec=0.5305, Rec=0.5056, F1=0.4876, Non-Fall: Acc=0.5432, W-Acc=0.1292, Prec=0.5710, Rec=0.5432, F1=0.4892
Global Test Metrics (Two Models) - Binary: Acc=0.6668, W-Acc=0.4446, Prec=0.7778, Rec=0.6668, F1=0.5336, Multiclass: Acc=0.5244, W-Acc=0.1267, Prec=0.5508, Rec=0.5244, F1=0.4884 

=== FEDAVG Round 5/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.8971, W-Acc=0.4933, Prec=0.9174, Rec=0.8971, F1=0.8990, Fall: Acc=0.8290, W-Acc=0.2211, Prec=0.8337, Rec=0.8290, F1=0.8282, Non-Fall: Acc=0.8261, W-Acc=0.1901, Prec=0.8282, Rec=0.8261, F1=0.8094
Average Client Metrics (Two Models) - Binary: Acc=0.8971, W-Acc=0.4933, Prec=0.9174, Rec=0.8971, F1=0.8990, Multiclass: Acc=0.8275, W-Acc=0.2056, Prec=0.8310, Rec=0.8275, F1=0.8188

Global Validation Metrics (Three Models) - Binary: Acc=0.6155, W-Acc=0.3997, Prec=0.5314, Rec=0.6155, F1=0.5483, Fall: Acc=0.4790, W-Acc=0.1483, Prec=0.3667, Rec=0.4790, F1=0.4001, Non-Fall: Acc=0.4970, W-Acc=0.1075, Prec=0.5109, Rec=0.4970, F1=0.4779
Global Validation Metrics (Two Models) - Binary: Acc=0.6155, W-Acc=0.3997, Prec=0.5314, Rec=0.6155, F1=0.5483, Multiclass: Acc=0.4880, W-Acc=0.1279, Prec=0.4388, Rec=0.4880, F1=0.4390

Global Test Metrics (Three Models) - Binary: Acc=0.6157, W-Acc=0.4083, Prec=0.4682, Rec=0.6157, F1=0.5179, Fall: Acc=0.5227, W-Acc=0.1277, Prec=0.5649, Rec=0.5227, F1=0.4944, Non-Fall: Acc=0.7516, W-Acc=0.1777, Prec=0.7225, Rec=0.7516, F1=0.7200
Global Test Metrics (Two Models) - Binary: Acc=0.6157, W-Acc=0.4083, Prec=0.4682, Rec=0.6157, F1=0.5179, Multiclass: Acc=0.6372, W-Acc=0.1527, Prec=0.6437, Rec=0.6372, F1=0.6072 

=== Running WEIGHTED_FEDAVG with Overlap = 0.0, Num Clients = 3 ===
=== WEIGHTED_FEDAVG Round 1/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.9095, W-Acc=0.5085, Prec=0.9103, Rec=0.9095, F1=0.9096, Fall: Acc=0.8286, W-Acc=0.2229, Prec=0.8362, Rec=0.8286, F1=0.8254, Non-Fall: Acc=0.8512, W-Acc=0.1921, Prec=0.8476, Rec=0.8512, F1=0.8330
Average Client Metrics (Two Models) - Binary: Acc=0.9095, W-Acc=0.5085, Prec=0.9103, Rec=0.9095, F1=0.9096, Multiclass: Acc=0.8399, W-Acc=0.2075, Prec=0.8419, Rec=0.8399, F1=0.8292

Global Validation Metrics (Three Models) - Binary: Acc=0.7047, W-Acc=0.3728, Prec=0.7828, Rec=0.7047, F1=0.7124, Fall: Acc=0.5226, W-Acc=0.1567, Prec=0.4486, Rec=0.5226, F1=0.4606, Non-Fall: Acc=0.5064, W-Acc=0.1119, Prec=0.5340, Rec=0.5064, F1=0.4888
Global Validation Metrics (Two Models) - Binary: Acc=0.7047, W-Acc=0.3728, Prec=0.7828, Rec=0.7047, F1=0.7124, Multiclass: Acc=0.5145, W-Acc=0.1343, Prec=0.4913, Rec=0.5145, F1=0.4747

Global Test Metrics (Three Models) - Binary: Acc=0.8440, W-Acc=0.4534, Prec=0.8881, Rec=0.8440, F1=0.8484, Fall: Acc=0.5459, W-Acc=0.1340, Prec=0.5851, Rec=0.5459, F1=0.5272, Non-Fall: Acc=0.7847, W-Acc=0.1871, Prec=0.7485, Rec=0.7847, F1=0.7556
Global Test Metrics (Two Models) - Binary: Acc=0.8440, W-Acc=0.4534, Prec=0.8881, Rec=0.8440, F1=0.8484, Multiclass: Acc=0.6653, W-Acc=0.1605, Prec=0.6668, Rec=0.6653, F1=0.6414 

=== WEIGHTED_FEDAVG Round 2/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.8805, W-Acc=0.5150, Prec=0.8066, Rec=0.8805, F1=0.8360, Fall: Acc=0.8684, W-Acc=0.2303, Prec=0.8744, Rec=0.8684, F1=0.8685, Non-Fall: Acc=0.8436, W-Acc=0.1910, Prec=0.8404, Rec=0.8436, F1=0.8240
Average Client Metrics (Two Models) - Binary: Acc=0.8805, W-Acc=0.5150, Prec=0.8066, Rec=0.8805, F1=0.8360, Multiclass: Acc=0.8560, W-Acc=0.2106, Prec=0.8574, Rec=0.8560, F1=0.8462

Global Validation Metrics (Three Models) - Binary: Acc=0.3339, W-Acc=0.1116, Prec=0.5727, Rec=0.3339, F1=0.1683, Fall: Acc=0.5616, W-Acc=0.1663, Prec=0.4781, Rec=0.5616, F1=0.5021, Non-Fall: Acc=0.5201, W-Acc=0.1107, Prec=0.5363, Rec=0.5201, F1=0.5072
Global Validation Metrics (Two Models) - Binary: Acc=0.3339, W-Acc=0.1116, Prec=0.5727, Rec=0.3339, F1=0.1683, Multiclass: Acc=0.5409, W-Acc=0.1385, Prec=0.5072, Rec=0.5409, F1=0.5047

Global Test Metrics (Three Models) - Binary: Acc=0.3348, W-Acc=0.1122, Prec=0.6946, Rec=0.3348, F1=0.1703, Fall: Acc=0.5415, W-Acc=0.1328, Prec=0.6235, Rec=0.5415, F1=0.5079, Non-Fall: Acc=0.7789, W-Acc=0.1870, Prec=0.7617, Rec=0.7789, F1=0.7524
Global Test Metrics (Two Models) - Binary: Acc=0.3348, W-Acc=0.1122, Prec=0.6946, Rec=0.3348, F1=0.1703, Multiclass: Acc=0.6602, W-Acc=0.1599, Prec=0.6926, Rec=0.6602, F1=0.6301 

=== WEIGHTED_FEDAVG Round 3/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.9711, W-Acc=0.5453, Prec=0.9730, Rec=0.9711, F1=0.9705, Fall: Acc=0.8653, W-Acc=0.2309, Prec=0.8780, Rec=0.8653, F1=0.8613, Non-Fall: Acc=0.8823, W-Acc=0.1941, Prec=0.8795, Rec=0.8823, F1=0.8722
Average Client Metrics (Two Models) - Binary: Acc=0.9711, W-Acc=0.5453, Prec=0.9730, Rec=0.9711, F1=0.9705, Multiclass: Acc=0.8738, W-Acc=0.2125, Prec=0.8788, Rec=0.8738, F1=0.8668

Global Validation Metrics (Three Models) - Binary: Acc=0.8673, W-Acc=0.4769, Prec=0.8821, Rec=0.8673, F1=0.8700, Fall: Acc=0.5525, W-Acc=0.1642, Prec=0.5056, Rec=0.5525, F1=0.4955, Non-Fall: Acc=0.5174, W-Acc=0.1106, Prec=0.5588, Rec=0.5174, F1=0.5125
Global Validation Metrics (Two Models) - Binary: Acc=0.8673, W-Acc=0.4769, Prec=0.8821, Rec=0.8673, F1=0.8700, Multiclass: Acc=0.5349, W-Acc=0.1374, Prec=0.5322, Rec=0.5349, F1=0.5040

Global Test Metrics (Three Models) - Binary: Acc=0.9582, W-Acc=0.5306, Prec=0.9602, Rec=0.9582, F1=0.9586, Fall: Acc=0.5415, W-Acc=0.1329, Prec=0.6218, Rec=0.5415, F1=0.5093, Non-Fall: Acc=0.8130, W-Acc=0.1918, Prec=0.8042, Rec=0.8130, F1=0.7882
Global Test Metrics (Two Models) - Binary: Acc=0.9582, W-Acc=0.5306, Prec=0.9602, Rec=0.9582, F1=0.9586, Multiclass: Acc=0.6773, W-Acc=0.1623, Prec=0.7130, Rec=0.6773, F1=0.6488 

=== WEIGHTED_FEDAVG Round 4/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.9887, W-Acc=0.5513, Prec=0.9889, Rec=0.9887, F1=0.9887, Fall: Acc=0.8859, W-Acc=0.2352, Prec=0.8883, Rec=0.8859, F1=0.8850, Non-Fall: Acc=0.8903, W-Acc=0.1961, Prec=0.8873, Rec=0.8903, F1=0.8800
Average Client Metrics (Two Models) - Binary: Acc=0.9887, W-Acc=0.5513, Prec=0.9889, Rec=0.9887, F1=0.9887, Multiclass: Acc=0.8881, W-Acc=0.2156, Prec=0.8878, Rec=0.8881, F1=0.8825

Global Validation Metrics (Three Models) - Binary: Acc=0.9527, W-Acc=0.5381, Prec=0.9544, Rec=0.9527, F1=0.9519, Fall: Acc=0.5735, W-Acc=0.1713, Prec=0.4799, Rec=0.5735, F1=0.5017, Non-Fall: Acc=0.5337, W-Acc=0.1107, Prec=0.5965, Rec=0.5337, F1=0.5355
Global Validation Metrics (Two Models) - Binary: Acc=0.9527, W-Acc=0.5381, Prec=0.9544, Rec=0.9527, F1=0.9519, Multiclass: Acc=0.5536, W-Acc=0.1410, Prec=0.5382, Rec=0.5536, F1=0.5186

Global Test Metrics (Three Models) - Binary: Acc=0.9907, W-Acc=0.5525, Prec=0.9909, Rec=0.9907, F1=0.9907, Fall: Acc=0.5511, W-Acc=0.1358, Prec=0.6182, Rec=0.5511, F1=0.5265, Non-Fall: Acc=0.8199, W-Acc=0.1909, Prec=0.7911, Rec=0.8199, F1=0.7973
Global Test Metrics (Two Models) - Binary: Acc=0.9907, W-Acc=0.5525, Prec=0.9909, Rec=0.9907, F1=0.9907, Multiclass: Acc=0.6855, W-Acc=0.1634, Prec=0.7046, Rec=0.6855, F1=0.6619 

=== WEIGHTED_FEDAVG Round 5/5 ===

Average Client Metrics (Three Models) - Binary: Acc=0.9790, W-Acc=0.5478, Prec=0.9798, Rec=0.9790, F1=0.9787, Fall: Acc=0.8926, W-Acc=0.2376, Prec=0.9030, Rec=0.8926, F1=0.8909, Non-Fall: Acc=0.8477, W-Acc=0.1937, Prec=0.8396, Rec=0.8477, F1=0.8249
Average Client Metrics (Two Models) - Binary: Acc=0.9790, W-Acc=0.5478, Prec=0.9798, Rec=0.9790, F1=0.9787, Multiclass: Acc=0.8702, W-Acc=0.2157, Prec=0.8713, Rec=0.8702, F1=0.8579

Global Validation Metrics (Three Models) - Binary: Acc=0.7296, W-Acc=0.4524, Prec=0.7284, Rec=0.7296, F1=0.6918, Fall: Acc=0.5545, W-Acc=0.1670, Prec=0.4745, Rec=0.5545, F1=0.4781, Non-Fall: Acc=0.3439, W-Acc=0.0632, Prec=0.4472, Rec=0.3439, F1=0.3412
Global Validation Metrics (Two Models) - Binary: Acc=0.7296, W-Acc=0.4524, Prec=0.7284, Rec=0.7296, F1=0.6918, Multiclass: Acc=0.4492, W-Acc=0.1151, Prec=0.4609, Rec=0.4492, F1=0.4097

Global Test Metrics (Three Models) - Binary: Acc=0.9871, W-Acc=0.5513, Prec=0.9873, Rec=0.9871, F1=0.9870, Fall: Acc=0.5539, W-Acc=0.1364, Prec=0.6211, Rec=0.5539, F1=0.5311, Non-Fall: Acc=0.5362, W-Acc=0.1216, Prec=0.5813, Rec=0.5362, F1=0.5256
Global Test Metrics (Two Models) - Binary: Acc=0.9871, W-Acc=0.5513, Prec=0.9873, Rec=0.9871, F1=0.9870, Multiclass: Acc=0.5451, W-Acc=0.1290, Prec=0.6012, Rec=0.5451, F1=0.5284

---------------------------------------------------------------------------128-------------------------------------------------------------------------------

