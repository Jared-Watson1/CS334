class Rectangle:
    x = None  # the x coordinate of the bottom left corner of the rectangle
    y = None  # the y coordinate of the bottom left corner of the rectangle
    l = None  # the length of the rectangle
    h = None  # the height of the circle

    def __init__(self, x=0, y=0, l=1, h=1):
        self.x = x
        self.y = y
        self.l = l
        self.h = h

    def area(self):
        """
        Compute the area of the rectangle

        Returns
        -------
        a : float
        """
        return self.l * self.h

    def perimeter(self):
        """
        Compute the perimeter of the rectangle

        Returns
        -------
        c : float
        """
        return 2 * self.l + 2 * self.h

    def contains_point(self, xc, yc):
        """
        Compute whether the rectangle contains the point (xc, yc).
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
        if self.x <= xc <= self.x + self.l and self.y <= yc <= self.y + self.h:
            return True
        return False


def main():
    # TODO print your name here
    print("Jared Watson")

    # Create an instance of the rectangle at 0,0, length=1 and height=1
    print("Creating Rectangle1 with bottom left at (0,0) with length=1 and height=1")
    rect1 = Rectangle()
    print("Area of Rectangle1", rect1.area())
    print("Perimeter of Rectangle1", rect1.perimeter())

    # TODO add a test cases here for 1I
    # true:
    print("Does Rectangle1 contain (0.5, 0.5): " + str(rect1.contains_point(0.5, 0.5)))

    # false:
    print("Does Rectangle1 contain (1.5, 1.5): " + str(rect1.contains_point(1.5, 1.5)))

    # Create an instance of the rectangle centered at 1,2 with length=3, height=4
    print("Creating Rectangle2 with bottom left at (1,2) with length=3 and height=4")
    rect2 = Rectangle(1, 2, 3, 4)
    print("Area of Rectangle2", rect2.area())
    print("Perimeter of Rectangle2", rect2.perimeter())


if __name__ == "__main__":
    main()
