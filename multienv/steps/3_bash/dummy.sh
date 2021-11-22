#!/bin/bash
valid=true
count=50
while [ $valid ]
do
echo "Acc:" $count
if [ $count -eq 60 ];
then
break
fi
((count++))
done
