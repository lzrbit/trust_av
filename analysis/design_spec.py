"""Author-confirmed design labels; see docs/data_schema.md for caveats."""
BLOCKS={'LC':'B1','SVM':'B2','HB':'B3','MB':'B4'}
SHEETS={'LC':'MAL','SVM':'AMB','HB':'HB','MB':'MB'}
SPECS={'LC':{'lateral_behaviour':['1m/s','3m/s','Fragmented','Abortion'],'acc_style':['cautious','mild','aggressive'],'design_distance':[5,15]},**{c:{'design_speed':[80,100,120],'design_braking':[-2,-5,-8],'design_distance':[5,15,25]} for c in ['SVM','HB','MB']}}
