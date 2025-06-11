class ChunkModel:
    def __init__(self, filename: str, content: str, chunknumber: int):
        self.filename = filename
        self.content = content
        self.chunknumber = chunknumber