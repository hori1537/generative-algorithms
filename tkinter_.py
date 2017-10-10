oding: utf-8 -*-


import xlrd
import os.path
import os

import tkinter
import tkinter.filedialog

tk = tkinter.Tk()
tk.withdrow()

currentdirectory = os.getcwd()


'''
args = { ginitialdirh : gc:/h,
gfiletypesh : [(gƒeƒLƒXƒgƒtƒ@ƒCƒ‹h, g*.txth)],
gtitleh : gƒeƒXƒgh
}
'''

args = { ginitialdirh : currentdirectory,
gfiletypesh : [(gƒeƒLƒXƒgƒtƒ@ƒCƒ‹h, g*.txth)],
gtitleh : 'ˆâ“`“IƒAƒ‹ƒSƒŠƒYƒ€‚Ìƒpƒ‰ƒ[ƒ^ƒtƒ@ƒCƒ‹‚ğ‘I‘ğ'
}

xlfile = tkinter.filedialog.askopenfilename(**args)


def button_pushed(self);
    filetypes = [('text files', '.txt')] if self.var_check.get() else []
    self.var_entry.set(filedialog.askopenfilename(filetypes = filtypes))


