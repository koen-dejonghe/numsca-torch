package scorch.function.loss

import ns.Region
import org.scalatest.{FlatSpec, Matchers}
import scorch.Variable
import torch.cpu.TH

class MeanSquaredErrorSpec extends FlatSpec with Matchers {
  "MSE" should "compute" in {

    Region.run { implicit r =>

      val x = Variable(ns.arange(min = 5, max = 13).reshape(4, 2))

      val target = Variable(ns.arange(min = 7, max = 11).reshape(1, 4))

      val mse = MeanSquaredError(x, target, sizeAverage = true, reduce = true)

      val error = mse.forward()

      println(error)
    }

//    error.backward()
//    println(x.grad)

  }

}
