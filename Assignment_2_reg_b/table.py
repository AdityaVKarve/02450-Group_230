import matplotlib.pyplot as plt
import numpy as np

# Sample data for the table
data = np.array([[1.23456789, 2.3456789, 3.456789],
                 [4.56789012, 5.67890123, 6.78901234],
                 [7.89012345, 8.90123456, 9.01234567]])

# Additional row for a common header
common_header_row = ["Common Header AB", "", "Common Header C"]

# Combine common header row with the existing data
data = np.vstack([common_header_row, data])

# Create a figure and axis
fig, ax = plt.subplots()

# Hide the axes
ax.axis('off')

# Create the table from a NumPy array
table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=None)

# Manually set column widths
col_widths = [0.2] * data.shape[1]  # Adjust the values as needed

# Set the column widths
table.auto_set_column_width(col=list(range(data.shape[1])))

# Set the font size for the table
table.auto_set_font_size(False)
table.set_fontsize(12)

# Display the table
plt.show()
