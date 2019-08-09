"""
Functions to compute distance to nearest point in matplotlib plot

"""
import numpy as np

class Picker(object):
    def __init__(self, space=None, data=None):
        assert space in [None, 2, 3]
        self.space = space  # dimensions of plot (2 or 3)
        self.data = data
        self.annotateLabel = None

    def setData(self, space, data):
        assert space in [2, 3]
        self.space = space
        self.data = data

    def setAction(self, action):
        # action is a subroutine that should be called when the
        # point is picked. In the original code, it was "annotate"
        # action will be called as self.action(closestIndex)
        self.action = action
        print('set action: ', self.action)
        
    def distance(self, point, event):
        """
        Return distance between mouse position and given data point

        Parameters
        ----------
        point : np.array
            np.array of shape (3,), with x,y,z in data coords
        event : MouseEvent:
            mouse event (which contains mouse position in .x and .xdata)
        Returns
        -------
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """

        if point.shape[0] not in [2, 3]:
            print("distance: point.shape is wrong: %s, must be (2,) or (3,)" % point.shape)
            return(None)
        if self.space == 3:
            # Project 3d data space to 2d data space
            x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], mpl.gca().get_proj())
        if self.space == 2:
           # x2, y2 = self.ax.transData.transform((point[0], point[1]))# Convert 2d data space to 2d screen space
            x2, y2 = (point[0], point[1])
        # x3, y3 = self.ax.transData.transform((x2, y2))
        # x = event.x  # for motion
        # y = event.y
        # print('event: ', event.ind, event.artist)
        xypos = event.artist.get_offsets()  # positions are in data space, not screen space here.
        ind = event.ind
        closest = 1e30
        clind = None
        for i in ind:  # there may be more than one point, so find closest
            dx = np.sqrt ((x2 - xypos[i, 0])**2 + (y2 - xypos[i, 1])**2)
            if dx < closest:
                closest = dx
                clind = i
        return closest


    def calcClosestDatapoint(self, event):
        """"
        Calculate which data point is closest to the mouse position.

        Parameters
        ----------
            event (MouseEvent) - mouse event (containing mouse position)
        
        Returns
        -------
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [self.distance(self.data[i][0:self.space], event) for i in range(self.data.shape[0])]
        return np.argmin(distances)

    # def annotatePlot(self, index):
    #     """Create popover label in 3d chart
    #
    #     Parameters
    #     ----------:
    #         index (int) - index (into points array X) of item which should be printed
    #     Returns
    #     -------
    #         Nothing
    #     """
    #     # If we have previously displayed another label, remove it first
    #     # if hasattr(self, 'annotateLabel'):
    #     #     self.annotatelabel.remove()
    #     if self.annotateLabel is not None:
    #         self.annotateLabel.remove()
    #     # Get data point from array of points X, at position index
    #     if self.space == 3:
    #         x2, y2, _ = proj3d.proj_transform(self.X[index, 0], self.X[index, 1], self.X[index, 2], self.ax.get_proj())
    #     if self.space == 2:
    #         #x2, y2 = self.ax.transData.transform((self.X[index, 0], self.X[index, 1]))# Convert 2d data space to 2d screen space
    #         x2, y2 = (self.X[index, 0], self.X[index, 1])
    #     # restructure text label:
    #     date = self.dFrame.iloc[index]['date']
    #     date = date.replace('/cell_00', 'C')
    #     date = date.replace('/slice_00', '-S')
    #     date = date.replace('_000', '_', 1)  # remove extra number from date
    #     date = date.replace('_', '')
    #     self.annotateLabel = mpl.annotate( "Cell \#%d %s %s" % (index, date,
    #                 self.dFrame.iloc[index]['cell_type']),
    #                 xy=(x2, y2), xytext=(-20, 20), zorder=1000,  # put in front!
    #                 textcoords='offset points', ha='right', va='bottom', fontsize=7,
    #                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9),
    #                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    #     self.figure.canvas.draw()

    def buttonPressEvent(self, event):
        """Event that is triggered when mouse is clicked. return index in array for closest data point"""
        print('buttonpress :', event.x, event.y)
        closestIndex = self.calcClosestDatapoint(event)
        print('buttonpress index', closestIndex)
        if closestIndex is None:
            return
        self.closestIndex = closestIndex
        self.action(closestIndex)
        
    def pickEvent(self, event):
        """Event that is triggered when mouse is clicked. return index in array for closest data point"""
        closestIndex = self.calcClosestDatapoint(event)
        print('pickevent index', closestIndex)
        if closestIndex is None:
            return
        self.closestIndex = closestIndex
        self.action(closestIndex)

    def onMouseMotion(self, event):
        """Movement jusr removes the annotation"""
        # print('mouse motion event: ', event)
        if self.annotateLabel is not None:
            self.annotateLabel.remove()
            self.annotateLabel = None