# Style guide for project contribution

---

## Class names:

All class names are written in **CamelCase**. That is: First letter of the class name is always an uppercase letter, and if the name is made of several words, the first letter of every new word is an uppercase letter as well. Words are not sepparated by any character.

> class MyEpicClass

---

## Method names:

All method names are written in **snake_case**. That is: First letter of the method is always a lowercase letter, and if the name is made of several words, the first letter of every new word is a lowercase letter as well. Words are separated by the character '_'.

> def my_epic_method():

---

## Attribute names:

Attribute names are written in lowercase letters. If the name is made of several words they all start with a lowercase letter as well. Words are not separated by any character 

**All attributes are private**

> self.__myepicattribute : int

---

## Property names:

All property names are written in **CamelCase**. That is: First letter of the property name is always an uppercase letter, and if the name is made of several words, the first letter of every new word is an uppercase letter as well. Words are not sepparated by any character.

Properties have *getters*, *setters* or both

Associated attributes follow attribute names convention

```python
    self.__myepicattribute : int

    @property
    def MyEpicAttribute(self) -> int:
        return self.__myepicattribute

    @MyEpicAttribute.setter
    def MyEpicAttribute(self, new_value : int) -> None:
        self.__myepicattribute = new_value

```

--- 

## Commiting to the project:

It's desirable (yet, not obligatory) to commit using [Conventional Commit Guide](https://www.conventionalcommits.org/en/v1.0.0/#summary).

Commits should be brief and not make a lot of changes. It's preferable to make several commits than to make one big commit.

---

## Branch management

When working on a new feature, it's desirable (yet, not obligatory) to create a new branch for that specific feature.

This is done in order to avoid conflicts.

---

## Import styling

Imports should import the whole package in order to avoid namespace conflicts. That is:

```python
    import myEpicPackage

    x = myEpicPackage.MyEpicClass()
```

instead of

```python
    from myEpicPackage import MyEpicClass

    x = MyEpicClass()
```

---

## Typing

Typing **must** be indicated in attributes and methods. Complex types should be indicated with the ```typing``` library.

For instance:

```python
    import typing

    x : typing.List[List[int]] = [[1,2,3], [4,5,6]]
    
    y : float = 3.14
```

```python
    import typing

    ...

    def cool_method(self, xx : int, yy : typing.Tuple[float]) -> None | bool:

        if xx != 0:
            return None
        
        else:
            return True

    def square_root(self, n : float) -> float:
        return math.sqrt(n)
```