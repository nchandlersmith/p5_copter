# p5_copter

You should have all the files that you need to run the quadcopter.
I have made extensive use of provided code from Udacity.
I implemented a reward function in task.py.
Also, I did tweak some hyperparameters.

It's task is to go to (0,0,100) starting at (0,0,10)
It looks like it is learning to stay near 0 in the x and y directions.
It has not learned to climb to 100 in z.

People that have finished the project report in the forum that they
spent much of their time working on the reward function. I believe
I have a pretty solid reward function, except for not controlling
the Euler angles which are pretty much all over the place, except
for psi, which stays at 0.

Based on what I have found in the forum, my next steps are:
1. Implement priority exerience playback
2. Tune learning rate
3. Implement real-time graph
4. Tune network architecture
5. Tune other hyper-params.

Any feedback on what I have would be appreciated.