package scorch

trait Function {
  def backward(gradOutput: Variable): Unit
  def forward(): Variable
}

