# Practice 1 git flow

### 1. Create an account on github.com if you don't have one, if you already have one then we're fine
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f1.png)

### 2. Configure your github account with SHH key, so as not to enter our username and password every time, we must investigate in Google
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f2.png)

### 3. Create a new repository called pratica_git_flow
### 4. Create a default README.md
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f3-f4.png)

### 5. Clone to our PC
```sh
git clone git@github.com:Luis-Alonso18/practica_git_flow.git
```
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f5.png)

### 6. Create a branch called development and make this branch the main one by default, this means that it will not be main
```sh
git branch development
git checkout development
```
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f6.png)

### 7. Making a change to the development branch in the README.md file could be "This is the development branch"
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f7.png)

### 8. Make a commit on the development branch
```sh
git commit -m "Commit description" 
```
### 9. Push the development branch
```sh
git push 
```
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f8-f9.png)

### 10. Create a branch called features
```sh
git branch features
```
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f10.png)

### 11. Making a change to the features branch in the README.md file can be "This is the features branch"
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f11.png)

### 12. Make the commit on this branch.
```sh
git commit -m "Commit description" 
```
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f12.png)
![alt text](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/practice_1_f13.png)
