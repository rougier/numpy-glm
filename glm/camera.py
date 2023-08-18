# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import numpy as np
from . import glm
from . ndarray import mat4
from . trackball import Trackball

class Camera():
    """
    Interactive trackball camera.

    This camera can be used for static or interactive rendering with mouse
    controlled movements. In this latter case, it is necessary to connect the
    camera to a matplotlib axes using the `connect` method and to provide an
    update function that will be called each time an update is necessary
    relatively to the new transform.

    In any case, the camera transformation is kept in the `Camera.transform`
    variable.
    """
    
    def __init__(self, mode="perspective", theta=0, phi=0, zdist=5.0, scale=1):
        """
        mode : str
          camera mode ("ortho" or "perspective")

        theta: float
          angle around z axis (degrees)

        phi: float
          angle around x axis (degrees)

        zdist : float
          Distance of the camera on the z-axis

        scale: float
          scale factor
        """
        
        self.aperture = 35
        self.aspect = 1
        self.near = 1
        self.far = 100
        self.mode = mode
        self.scale = scale
        self.zoom = 1
        self.zoom_max = 5.0
        self.zoom_min = 0.1
        self.transform = mat4()
        
        if mode == "ortho":
            self.proj = glm.ortho(-1,+1,-1,+1, self.near, self.far)
            self.trackball = None
            self.transform[...] = self.proj
        else:
            self.trackball = Trackball(theta, phi)
            self.proj = glm.perspective(
                self.aperture, self.aspect, self.near, self.far)
            self.view = glm.translate((0, 0, -zdist)) @ glm.scale((scale,scale,scale))
            self.transform[...] = self.proj @ self.view @ self.trackball.model.T
        self.updates = {"motion"  : [],
                        "scroll"  : [],
                        "press"   : [],
                        "release" : []}

    def update(self, event):
        """
        Update all connected objects
        """

        for update in self.updates[event]:
            update(self.transform)
        
        
    def connect(self, axes, event, update):
        """
        axes : matplotlib.Axes
           Axes where to connect this camera to

        event: string
           Which event to connect to (motion, scroll, press, release)
        
        update: function(transform)
           Function to be called with the new transform to update the scene
           (transform is a 4x4 matrix).
        """
        
        self.figure = axes.get_figure()
        self.axes = axes
        # self.update = update
        if update not in self.updates[event]:
            self.updates[event].append(update)
        
        self.mouse = None
        self.cidscroll = self.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

        def format_coord(*args):
            phi = self.trackball.phi
            theta = self.trackball.theta
            return "Θ : %.1f, ɸ: %.1f" % (theta, phi)
        if self.trackball is not None:
            self.axes.format_coord = format_coord
                

        
    def on_scroll(self, event):
        """
        Scroll event for zooming in/out
        """
        if event.inaxes != self.axes:
            return
        
        if event.button == "up":
            self.zoom  = max(0.9*self.zoom, self.zoom_min)
        elif event.button == "down":
            self.zoom = min(1.1*self.zoom, self.zoom_max)
        self.axes.set_xlim(-self.zoom, self.zoom)
        self.axes.set_ylim(-self.zoom, self.zoom)
        self.update("scroll")
        self.figure.canvas.draw()

        
    def on_press(self, event):
        """
        Press event (initiate drag)
        """
        if event.inaxes != self.axes:
            return
        
        self.mouse = event.button, event.xdata, event.ydata
        self.update("press")
        self.figure.canvas.draw()
        
        
    def on_motion(self, event):
        """
        Motion event to rotate the scene
        """
        if self.mouse is None:            return
        if event.inaxes != self.axes:     return
        if self.trackball is None:        return
        
        button, x, y = event.button, event.xdata, event.ydata
        dx, dy = x-self.mouse[1], y-self.mouse[2]
        self.mouse = button, x, y
        self.trackball.drag_to(x, y, dx, dy)        
        self.transform[...] = self.proj @ self.view @ self.trackball.model.T
        self.update("motion")
        self.figure.canvas.draw()

        
    def on_release(self, event):
        """
        Release event (end of drag)
        """
        self.mouse = None
        self.update("release")
        self.figure.canvas.draw()

        
    def disconnect(self):
        """
        Disconnect camera from the axes
        """
        self.figure.canvas.mpl_disconnect(self.cidscroll)
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)
