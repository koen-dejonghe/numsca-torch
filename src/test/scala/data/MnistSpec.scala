package data

import ns.LongTensor
import org.scalatest.{FlatSpec, Matchers}
import scorch.Function._
import scorch.module.Linear
import scorch.{Module, Variable}
import scorch.optimizer.SGD

class MnistSpec extends FlatSpec with Matchers {

  "Mnist" should "simple feed nn" in {

    val numSamples = 32
    val numFeatures = 28 * 28
    val numClasses = 10

    ns.setSeed(231)
    case class Net() extends Module {
      val fc1 = Linear(numFeatures, 100)
      val fc2 = Linear(100, numClasses)
      override def forward(x: Variable): Variable =
        x ~> fc1 ~> relu ~> fc2
    }

    val net = Net()

    val trainingSet = new MnistDataLoader("train", numSamples)
    val devSet = new MnistDataLoader("train", numSamples)

    val optimizer = SGD(net.parameters, lr = 0.08)

    for (epoch <- 1 to 100) {

      var avgLoss = 0.0
      var avgAccuracy = 0.0
      var count = 0

      trainingSet.foreach {
        case (y, x) =>
          count += 1

          val targetAsLong = LongTensor(y)

          net.zeroGrad()
          val output = net(Variable(x))

          val guessed = ns.argmax(output, axis = 1)
          val accuracy = ns.sum(guessed == y) / numSamples
          avgAccuracy += accuracy

          val loss = crossEntropy(output, targetAsLong)
          avgLoss += loss.value(0)

          loss.backward()
          optimizer.step()
      }
      println(s"$epoch: loss: ${avgLoss / count} accuracy: ${avgAccuracy / count}")
    }
  }
}
