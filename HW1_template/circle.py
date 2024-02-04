import math


class Circle:
    x = None  # the x coordinate of the circle's center
    y = None  # the y coordinate of the circle's center
    r = None  # the radius of the circle

    def __init__(self, x=0, y=0, r=1):
        self.x = x
        self.y = y
        self.r = r

    def area(self):
        """
        Compute the area of the circle

        Returns
        -------
        a : float
        """
        pi = math.pi

        return pi * (self.r**2)

    def circumference(self):
        """
        Compute the circumference of the circle

        Returns
        -------
        c : float
        """
        pi = math.pi
        return 2 * pi * self.r

    def contains_point(self, xc, yc):
        """
        Compute whether the circle contains the point (xc, yc).
        For the corner case of the surface area, it should return true.

        Parameters
        ----------
        xc : float
            x coordinate
        yc : float
            y coordinate

        Returns
        -------
        b : boolean
        """
        distance = math.sqrt((xc - self.x) ** 2 + (yc - self.y) ** 2)

        if distance <= self.r:  # inside of circle or on edge
            return True
        return False


def main():
    # TODO print your name here
    print("Jared Watson")

    # Create an instance of the circle centered at 0,0 with radius of 1
    print("Creating Circle1 centered at (0,0) with radius 1")
    circle1 = Circle()
    print("Area of Circle1", circle1.area())
    print("Circumference of Circle1", circle1.circumference())

    # TODO add test cases here for 1D.
    # true:
    print("Does Circle1 contain (0.5, 0.5): " + str(circle1.contains_point(0.5, 0.5)))
    # false:
    print("Does Circle1 contain (3, 3): " + str(circle1.contains_point(1.5, 1.5)))

    # Create an instance of the circle centered at 1,2 with radius of 3
    print("Creating Circle2 centered at (1,2) with radius 3")
    circle2 = Circle(1, 2, 3)
    print("Area of Circle2", circle2.area())
    print("Circumference of Circle2", circle2.circumference())


if __name__ == "__main__":
    main()
