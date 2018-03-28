import tkinter
import config as cfg
from game import Board

class graphicBoard(Board):
    def __init__(self):
        super(graphicBoard,self).__init__()

        self.root = tkinter.Tk()
        self.root.geometry('600x600')
        self.block_size = 500 // (cfg.board_size - 1)
        self.board_width = self.block_size * (cfg.board_size - 1)
        self.piece_size = min(30,self.block_size / 1.5)

        self.canvas=tkinter.Canvas(self.root,width=600,height=600,bg='gray')
        self.canvas.pack()

        for i in range(cfg.board_size):
            self.canvas.create_line(50 + i * self.block_size, 50, 50 + i * self.block_size, 50 + self.board_width,
                                    width=2)
            self.canvas.create_line(50, 50 + i * self.block_size, 50 + self.board_width, 50 + i * self.block_size,
                                    width=2)

        self.canvas.bind('<Button-1>', self.draw_next_piece_by_click)
        self.root.mainloop()

    def draw_piece(self,x,y):
        if self.current_player:
            color = "white"
        else:
            color = "black"
        pix_x_position = 50 + x * self.block_size - self.piece_size / 2
        pix_y_position = 50 + y * self.block_size - self.piece_size / 2
        self.canvas.create_oval(pix_x_position, pix_y_position, pix_x_position + self.piece_size,
                                pix_y_position + self.piece_size, fill=color)
        self.move(x + y * cfg.board_size)

    def draw_next_piece_by_click(self,event):
        x = event.x - 50
        y = event.y - 50
        print(x,y)
        if x < 0 or y < 0 or x > self.board_width or y > self.board_width:
            # the click out of the board
            return
        x = round(x / self.block_size)
        y = round(y / self.block_size)
        self.draw_piece(x,y)




graphicBoard()