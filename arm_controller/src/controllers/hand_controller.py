# import impire_hand_wr_tactile as hand_wr

class HandController:
    def __init__(self, hand, type = 'gripper'):
        self.hand = hand
        self.type = type

    # def read_angles(self):
    #     return hand_wr.read6(self.client, "angleAct")
    #
    # def read_forces(self):
    #     return hand_wr.read6(self.client, "forceAct")
    #
    def read_width(self):
        if self.type == 'gripper':
            return self.hand.read_once().width
        else:
            print('No gripper')
            return 0

    def release_gripper(self, width = 0.09, speed = 0.03):
        if self.type == 'gripper':
            self.hand.move(width, speed)
        return True

    def move_gripper(self, width, speed):
        if self.type == 'gripper':
            self.hand.move(width, speed)
        return True

    def grasp(self, width, speed, force = 10.0):
        if self.type == 'gripper':
            print("gripper grasp")
            self.hand.grasp(width, speed, force)
            return True
        else:
            print('No gripper')
            return 0