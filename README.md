
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