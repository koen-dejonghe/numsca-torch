name := "numsca-torch"

version := "0.1"

scalaVersion := "2.12.6"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  // "me.tongfei" % "jtorch-cpu" % "0.3.0-SNAPSHOT",
  "be.botkop" % "torch-cpu" % "0.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "3.0.5" % Test
)

fork in Test := true
javaOptions in Test ++= Seq("-Djava.library.path=/Users/koen/projects/ATen/target/lib")