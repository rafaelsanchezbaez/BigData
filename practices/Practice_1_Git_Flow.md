#practice 1 git flow

1. Create an account on github.com if you don't have one, if you already have one then we're fine
(Imagen)

2. Configure your github account with SHH key, so as not to enter our username and password every time, we must investigate in Google
(Imagen)

3. Create a new repository called pratica_git_flow
4. Create a default README.md
(Imagen)

5. Clone to our PC
MD
```
git clone git@github.com:Luis-Alonso18/Data_Mining.git
```
(Image)
6. Create a branch called development and make this branch the main one by default, this means that it will not be main
MD
```
git branch development
git checkout development
```
(Imagen)

7. Making a change to the development branch in the README.md file could be "This is the development branch"
(Imagen)

8. Make a commit on the development branch
MD
```
git commit -m "Commit description" 
```
9. Push the development branch
MD
```
git push 
```
(Imagen)

10. Create a branch called features
MD
```
git branch features
```
(Imagen)

11. Making a change to the features branch in the README.md file can be "This is the features branch"
(Image)

12. Make the commit on this branch.
MD
```
git commit -m "Commit description" 
```
(Image)