# Boundary analysis fixes

- [x] ** Add command to skip .xlsx files for tracking coordinates.**

- [x] ** Linear arena, short walls are not icluded in edge zone.**

- [x] ** Linear track occupancy across short wall is problematic because of 5 bins instead of 4 spanning the width. Figure out a way to fix this.** *3 wall bins are included in edge zone, 2 center bins are included in center zone.*

- [x] ** Proprtions look off. Run on entire dataset and check if they yield the same %s.** * check code algo if that doesn't work.* *Biased by the small dataset used. Net result with 70-30 distribution still holds true.*

- [x] <Don't use open arena for boundary analysis. Occupany is very unevenly distributed between zones.>

- [x] ** Add other parameters <ONLY after confirming the code works>.**