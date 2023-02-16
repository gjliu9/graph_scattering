
k=5
python mGST_classification_ENZYMES.py --variation $k --sample umin --gstcoe_norml MS
python mGST_classification_ENZYMES.py --variation $k --sample umin --gstcoe_norml none
python mGST_classification_ENZYMES.py --variation $k --sample umin --gstcoe_norml MS --feature_choosen 3
python mGST_classification_ENZYMES.py --variation $k --sample umin --gstcoe_norml none --feature_choosen 3
python mGST_classification_ENZYMES.py --variation $k --sample umax --gstcoe_norml MS
python mGST_classification_ENZYMES.py --variation $k --sample umax --gstcoe_norml none
python mGST_classification_ENZYMES.py --variation $k --sample umax --gstcoe_norml MS --feature_choosen 3
python mGST_classification_ENZYMES.py --variation $k --sample umax --gstcoe_norml none --feature_choosen 3
