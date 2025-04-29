
The noun list
https://www.desiquintans.com/nounlist

The pluralizer
https://github.com/plurals/pluralize
https://pypi.org/project/pluralizer/

- behaves correctly when the plural is the same as the singular
- Just adds an 's' for singularia tantum nouns (so grammar model correctly identifies it as wrong)

```python
print("Homework", end=" -> ")
print(pluralizer.pluralize("homework"))
print("Furniture", end=" -> ")
print(pluralizer.pluralize("furniture"))

# Homework -> homework
# Furniture -> furnitures
```

Inflections
https://github.com/jaraco/inflect

## Visualizations

### General statistics of the nouns

#### a) Make the installation script executable.

```bash
chmod +x install_dependencies.sh
```

#### b) Install the necessary dependencies.

```bash
./install_dependencies.sh
```

#### c) Activate the virtual environment:

```bash
source venv/bin/activate
```

#### d) Run the visualization script:

```bash
python visualizations.py
```

#### e) When prompted, enter the column name you want to visualize.

#### f) Deactivate the virtual environment.

```bash
deactivate
```

### K-means 

Repeat steps a) - c)

#### d) Run the visualization script:

```bash
python kmeans_visualization.py
```

Repeat steps e) & f)
