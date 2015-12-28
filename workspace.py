import csv
from sets import Set

file_handle=open('Farhan_Sample Dishname file for vegnonveg indicator.csv','rt')
reader=csv.DictReader(file_handle)

c1=0
c2=0
c3=0
c4=0
veg=[]
nveg=[]
both=[]

for line in reader:
	if line['veg_ind']==line['nonveg_ind']:
		# print line['dish name']
		c3=c3+1
		both.append(line['dish name'])
	elif line['veg_ind']=='1':
		c1=c1+1
		veg.append(line['dish name'])
	elif line['nonveg_ind']=='1':
		c2=c2+1
		nveg.append(line['dish name'])
	c4=c4+1

veg=Set(veg)
nveg=Set(nveg)
both=Set(both)

print nveg & both

print c1,c2,c3,c4

# Set(['Dum Ki Biryani', 'Chicken Wings', 'bbq chicken wings', 'Chicken Stroganoff', 'Garlic Bread', 
# 	'Veg Sampler', 'malai kofta', 'murg tikka masala', 'Blueberry Cheesecake', 'Veg Mezze Platter', '
# Strawberry Seduction', 'American Breakfast', 'Fish Fry', 'loaded nachos', 'chicken biryani', 'Fruit Beer', 
# 'dahi kabab', 'Chilli Hot Dog', 'Pesto Pasta', 
# 	'Hot Chocolate', 'Grilled Chicken Breast', 'Red Velvet Cupcake', 'Espresso Pan Fried Chicken Breast', 'galouti kebab