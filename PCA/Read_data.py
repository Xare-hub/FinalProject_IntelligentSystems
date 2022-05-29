import pandas as pd

# Each column units are kg in a m^3 mixture
# Except the Age column, which is in days
# and the Compressive stregth column, which is in Megapascals

# Read data as a dataframe with Pandas
data = pd.read_excel(r"C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\10mo Semestre\SI\FinalProject\Datasets\Concrete\Concrete_Data.xls")

# Transform Pandas dataframe to numpy array
np_data = data.to_numpy()

print(data)
print(np_data.shape)
