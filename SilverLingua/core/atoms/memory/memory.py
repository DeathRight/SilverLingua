class Memory:
  content: str

  def __init__(self, content: str) -> None:
    self.content = content

  def __str__(self) -> str:
    return self.content

  def __repr__(self) -> str:
    return f"Memory({self.content})"

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Memory):
      return NotImplemented
    return self.content == other.content