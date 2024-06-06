# model_cloud_deployment

Mission :
application mobile qui permettrait aux utilisateurs de prendre en photo un fruit et d'obtenir des informations sur ce fruit
-> mettre en place une première version du moteur de classification des images de fruits
-> architecture Big Data nécessaire

Données :
Kaggle -> https://www.kaggle.com/datasets/moltean/fruits

Notebook de départ :
Première approche dans un environnement Big Data AWS EMR, à partir d’un jeu de données constitué des images de fruits et des labels associés (cf données kaggle).
Le notebook réalisé par l’alternant servira de point de départ pour construire une partie de la chaîne de traitement des données.

Vous êtes donc chargé de vous approprier les travaux réalisés par l’alternant et de compléter la chaîne de traitement.
Il n’est pas nécessaire d’entraîner un modèle pour le moment.
L’important est de mettre en place les premières briques de traitement qui serviront lorsqu’il faudra passer à l’échelle en termes de volume de données !


Install java:
dans terminal : brew install openjdk@11

Set Java environment variables:
After installing Java, you'll need to set the JAVA_HOME environment variable to point to the location where Java is installed. You can add the following lines to your ~/.bash_profile, ~/.zshrc, or ~/.profile file, depending on the shell you are using : 
export JAVA_HOME=/opt/homebrew/opt/openjdk@11
export PATH=$JAVA_HOME/bin:$PATH

Créer un compte AWS