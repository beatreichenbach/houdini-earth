from typing import cast

import hou


class State:
    def __init__(self, state_name: str, scene_viewer: hou.SceneViewer) -> None:
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.guide = False

    def onEnter(self, kwargs: dict[str, object]) -> None:
        node = cast(hou.SopNode, kwargs['node'])
        parm = node.parm('guide')
        if parm is not None:
            self.guide = bool(parm.eval())
            parm.set(1)

    def onExit(self, kwargs: dict[str, object]) -> None:
        node = cast(hou.SopNode, kwargs['node'])
        parm = node.parm('guide')
        if parm is not None:
            parm.set(self.guide)


def createViewerStateTemplate() -> hou.ViewerStateTemplate:
    state_typename = kwargs['type'].definition().sections()['DefaultState'].contents()  # noqa: F821 # ty: ignore[unresolved-reference]
    state_label = 'Earth Mesh'
    state_category = hou.sopNodeTypeCategory()
    state_icon = kwargs['type'].icon()  # noqa: F821 # ty: ignore[unresolved-reference]

    template = hou.ViewerStateTemplate(state_typename, state_label, state_category)
    template.bindFactory(State)
    template.bindIcon(state_icon)

    return template
